"""DiscreteMemory 离线评测入口（在项目根目录运行）。

    python eval/run_eval.py
        真实嵌入模型 + 真实 LLM，跑全部四类评测

    python eval/run_eval.py --suites extraction,consent
        只跑不需要模型的部分，几秒钟完成

    python eval/run_eval.py --embedder hash --llm stub
        替身模式，只用来检查评测脚本本身，结果不代表系统表现

    python eval/run_eval.py --compare /absolute/path/baseline /absolute/path/after
        对比两次运行，输出前后对比表

结果默认写到项目 eval/results/<name>/：report.md（结果表）、summary.json（全部指标）、
details/*.jsonl（逐条明细，用于错误分析）。
未指定 --name 时，目录名包含实际评测内容、运行模式与日期时间。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib import metadata as package_metadata
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from pathlib import Path
from types import SimpleNamespace

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent
for _path in (str(PROJECT_ROOT), str(EVAL_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

EVALUATION_PROTOCOL = "discretememory-eval-1.4"

def _local_path(value):
    """评测脚本在 WSL 中执行时，兼容 Windows 盘符路径。"""
    value = str(value)
    if os.name != "nt" and len(value) >= 3 and value[1] == ":" and value[2] in "/\\":
        mount = Path("/mnt") / value[0].lower()
        if mount.is_dir():
            return mount / value[3:].replace("\\", "/")
    return Path(value).expanduser()

def _workspace():
    explicit = os.getenv("DISCRETE_MEMORY_EVAL_HOME")
    if explicit:
        path = _local_path(explicit)
        return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()
    return EVAL_DIR

WORKSPACE = _workspace()
TEMP_ROOT = WORKSPACE / "tmp"
RESULTS_ROOT = WORKSPACE / "results"

def _now():
    zone = os.getenv("DISCRETE_MEMORY_EVAL_TIMEZONE", "UTC")
    try:
        return datetime.now(ZoneInfo(zone))
    except ZoneInfoNotFoundError:
        return datetime.now(timezone.utc)

def _component(value):
    """确保名称只表示一个目录（包括 Windows 环境），拒绝越界路径。"""
    value = str(value)
    if not value or value in {".", ".."} or any(c in value for c in "/\\:") or Path(value).is_absolute():
        raise ValueError(f"expected one directory name: {value!r}")
    return value

def _create_result_directory(root: Path, name: str, *, automatic: bool) -> Path:
    """创建本次输出目录，不覆盖之前的运行结果。"""
    name = _component(name)
    root.mkdir(parents=True, exist_ok=True)
    number = 1
    while True:
        candidate = root / (name if number == 1 else f"{name}_{number:02d}")
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            if not automatic:
                raise ValueError(f"结果目录已存在，请更换 --name：{candidate}") from None
            number += 1


def _within(target, root):
    target, root = Path(target).resolve(), Path(root).resolve()
    if target == root or root not in target.parents:
        raise ValueError(f"path leaves isolated scenario directory: {target}")
    return target

def source_fingerprint(paths, root):
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()

def implementation_metadata(config):
    runtime_files = [p for p in PROJECT_ROOT.rglob("*.py")
                     if p.relative_to(PROJECT_ROOT).parts[0] not in {"eval", ".git", ".idea", ".venv", "venv", "__pycache__"}]
    versions = {}
    for name in ("numpy", "faiss-cpu", "faiss-gpu", "torch", "sentence-transformers", "requests", "python-dotenv"):
        try:
            versions[name] = package_metadata.version(name)
        except package_metadata.PackageNotFoundError:
            versions[name] = None
    settings = {name: getattr(config, name, None) for name in (
        "VECTOR_DIM", "BASE_MAX_DISTANCE", "MEMORY_MAX_DISTANCE", "BASE_RECALL_K",
        "MEMORY_RECALL_K", "SHORT_TERM_LIMIT", "CHUNK_SIZE", "OVERLAP_SENTENCE")}
    return {
        "protocol": EVALUATION_PROTOCOL,
        "runtime_sha256": source_fingerprint(runtime_files, PROJECT_ROOT),
        "evaluator_sha256": source_fingerprint(EVAL_DIR.glob("*.py"), EVAL_DIR),
        "runtime_config": settings, "dependencies": versions,
    }

ALL_SUITES = ["extraction", "consent", "lifecycle", "latency"]
LANG_LABEL = {"zh": "中文", "en": "英文", "mix": "中英混合"}

SUITE_LABELS = {
    "extraction": "规则抽取与槽位识别",
    "consent": "规则写入准入",
    "lifecycle": "记忆生命周期与召回",
    "latency": "聊天延迟",
}


def _default_run_name(suites, embedder_kind, llm_kind, *, latency_repeats):
    """描述实际执行的评测套件，以及这些套件实际使用的模型。"""
    scope = "全部评测" if set(suites) == set(ALL_SUITES) else "+".join(
        SUITE_LABELS[suite] for suite in suites
    )
    modes = []
    if any(suite in suites for suite in ("lifecycle", "latency")):
        modes.append({"real": "真实Embedding", "hash": "模拟Embedding"}[embedder_kind])
    if any(suite in suites for suite in ("latency",)):
        modes.append({"real": "真实LLM", "stub": "模拟LLM"}[llm_kind])
    parts = [scope, "+".join(modes) if modes else "规则评测"]
    if "latency" in suites:
        parts.append(f"延迟{latency_repeats}轮")
    parts.append(_now().strftime("%Y-%m-%d_%H-%M-%S"))
    return "_".join(parts)


LIFECYCLE_CHECKS = [
    # (检查项, 中文名, 是否依赖嵌入模型的语义能力)
    ("sql_state", "SQLite 状态正确", False),
    ("vector_consistent", "向量库与 SQLite 一致", False),
    ("history_correct", "版本历史正确", False),
    ("slot_lookup_correct", "按槽位读取返回正确值", False),
    ("no_stale_leak", "全量检索中无旧值或已删值", False),
    ("purge_sql_gone", "清除后 SQLite 查不到", False),
    ("purge_vector_gone", "清除后向量库没有任何版本", False),
    ("purge_export_clean", "清除后导出数据不含该值", False),
    ("purge_db_bytes_clean", "指定数据库文件未检出样例字符串", False),
    ("purge_snapshot_bytes_clean", "指定向量快照未检出样例字符串", False),
    ("candidate_created", "用户原话生成了候选记忆", False),
    ("candidate_hidden", "确认前候选不可被检索", False),
    ("candidate_confirmed_visible", "确认后可被检索", False),
    ("candidate_rejected_hidden", "拒绝后仍不可被检索", False),
    ("recovered_after_restart", "重启后状态正确", False),
    ("recovered_after_index_loss", "向量库丢失后能从 SQLite 恢复", False),
    ("semantic_hit", "语义检索命中新值（top-k）", True),
]


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def norm_value(value) -> str:
    return "".join(str(value).lower().split()).strip("，。！？,.!?：:")


def ratio(numerator, denominator):
    return None if not denominator else numerator / denominator


def pct(value) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def ms(value) -> str:
    if value is None:
        return "—"
    return f"{value / 1000:.2f} s" if value >= 1000 else f"{value:.0f} ms"


def percentile(values, p):
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(p / 100 * len(ordered)))
    return ordered[rank - 1]


def prf(tp, fp, fn):
    precision = ratio(tp, tp + fp)  # 一条都没抽出时精确率无定义，报告里显示为“—”
    recall = ratio(tp, tp + fn)
    if tp == 0:
        f1 = 0.0 if (fp + fn) else None
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def md_table(headers, rows) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def load_config(required: bool):
    """读取项目 Config；不需要模型时允许回退到默认值。"""
    try:
        from config import Config  # noqa: WPS433
        return Config
    except Exception as exc:  # pragma: no cover - 取决于本地 .env
        if required:
            raise
        print(f"[提示] 读取 config.py 失败，使用默认参数：{exc}")
        return SimpleNamespace(
            VECTOR_DIM=512, BASE_MAX_DISTANCE=1.2, MEMORY_MAX_DISTANCE=1.5,
            BASE_RECALL_K=3, MEMORY_RECALL_K=3, SHORT_TERM_LIMIT=5,
            CHUNK_SIZE=300, OVERLAP_SENTENCE=1,
            EMBED_MODEL_PATH="(未读取)", LLM_MODEL="(未读取)",
        )


class Models:
    """按需加载模型，只加载一次。"""

    def __init__(self, embedder_kind: str, llm_kind: str):
        self.embedder_kind = embedder_kind
        self.llm_kind = llm_kind
        self.config = load_config(required=(embedder_kind == "real" or llm_kind == "real"))
        self._embedder = None
        self._llm = None

    @property
    def embedder(self):
        if self._embedder is None:
            if self.embedder_kind == "real":
                from models.embedding import EmbeddingModel
                self._embedder = EmbeddingModel(self.config.EMBED_MODEL_PATH)
            else:
                from fakes import HashEmbedder
                self._embedder = HashEmbedder(self.config.VECTOR_DIM)
        return self._embedder

    def chat_llm(self):
        if self.llm_kind == "real":
            return self._real_llm()
        from fakes import StubChatLLM
        return StubChatLLM()


    def _real_llm(self):
        if self._llm is None:
            from models.llm import LLMModel
            self._llm = LLMModel(
                self.config.OLLAMA_URL,
                self.config.LLM_MODEL,
                timeout=self.config.LLM_TIMEOUT_SECONDS,
            )
        return self._llm

    def describe(self) -> dict:
        embed = (
            Path(str(self.config.EMBED_MODEL_PATH)).name
            if self.embedder_kind == "real" else
            "HashEmbedder（替身）" if self.embedder_kind == "hash" else "不使用"
        )
        llm = {
            "real": str(getattr(self.config, "LLM_MODEL", "?")),
            "stub": "替身（对话固定回答）",
            "none": "不使用",
        }[self.llm_kind]
        return {"embedder": embed, "llm": llm}


# ---------------------------------------------------------------------------
# 1. 记忆抽取 + 回忆问题路由
# ---------------------------------------------------------------------------

def run_extraction(cases_dir: Path):
    from memory.extractor import DiscreteMemoryExtractor

    extractor = DiscreteMemoryExtractor()
    details = []
    for case in load_jsonl(cases_dir / "extraction.jsonl"):
        predicted = [{"slot": f.slot, "value": f.value} for f in extractor.extract(case["text"])]
        expected_set = {(e["slot"], norm_value(e["value"])) for e in case["expected"]}
        predicted_set = {(p["slot"], norm_value(p["value"])) for p in predicted}
        expected_slots = {e["slot"] for e in case["expected"]}
        predicted_slots = {p["slot"] for p in predicted}
        details.append({
            **case,
            "predicted": predicted,
            "tp": len(expected_set & predicted_set),
            "fp": len(predicted_set - expected_set),
            "fn": len(expected_set - predicted_set),
            "slot_tp": len(expected_slots & predicted_slots),
            "slot_fp": len(predicted_slots - expected_slots),
            "slot_fn": len(expected_slots - predicted_slots),
            "correct": expected_set == predicted_set,
        })

    def aggregate(rows):
        tp = sum(r["tp"] for r in rows)
        fp = sum(r["fp"] for r in rows)
        fn = sum(r["fn"] for r in rows)
        precision, recall, f1 = prf(tp, fp, fn)
        _, _, slot_f1 = prf(
            sum(r["slot_tp"] for r in rows),
            sum(r["slot_fp"] for r in rows),
            sum(r["slot_fn"] for r in rows),
        )
        return {"n": len(rows), "precision": precision, "recall": recall, "f1": f1, "slot_f1": slot_f1}

    positives = [r for r in details if r["kind"] != "negative"]
    negatives = [r for r in details if r["kind"] == "negative"]
    groups = {}
    for lang in ("zh", "en", "mix"):
        rows = [r for r in positives if r["lang"] == lang]
        if rows:
            groups[lang] = aggregate(rows)
        for split in ("dev", "heldout"):
            subset = [r for r in rows if r["split"] == split and r["kind"] == "single"]
            if subset and lang != "mix":
                groups[f"{lang}.{split}"] = aggregate(subset)
        compound = [r for r in rows if r["kind"] == "compound"]
        if compound:
            groups[f"{lang}.compound"] = aggregate(compound)
    negative_fp = {}
    for lang in ("zh", "en"):
        rows = [r for r in negatives if r["lang"] == lang]
        negative_fp[lang] = {
            "n": len(rows),
            "fp_rate": ratio(sum(1 for r in rows if r["predicted"]), len(rows)),
            "by_category": dict(Counter(r["category"] for r in rows if r["predicted"])),
        }

    # 回忆问题路由：问题能否被识别为正确的记忆槽位
    routing_details = []
    for case in load_jsonl(cases_dir / "recall_routing.jsonl"):
        predicted = extractor.infer_recall_slots(case["text"])
        routing_details.append({**case, "predicted_slots": predicted,
                                "correct": sorted(predicted) == sorted(case["expected_slots"])})
    routing = {}
    for lang in ("zh", "en"):
        for split in (None, "dev", "heldout"):
            rows = [r for r in routing_details if r["lang"] == lang and (split is None or r["split"] == split)]
            key = lang if split is None else f"{lang}.{split}"
            routing[key] = {"n": len(rows), "accuracy": ratio(sum(r["correct"] for r in rows), len(rows))}

    summary = {"groups": groups, "negative": negative_fp, "routing": routing}
    return summary, {"extraction": details, "recall_routing": routing_details}


# ---------------------------------------------------------------------------
# 2. 同意判定
# ---------------------------------------------------------------------------

def run_consent(cases_dir: Path):
    from memory.extractor import DiscreteMemoryExtractor
    from product.write_policy import ConsentAwareWritePolicy

    extractor = DiscreteMemoryExtractor()
    policy = ConsentAwareWritePolicy()
    details = []
    for case in load_jsonl(cases_dir / "consent.jsonl"):
        facts = extractor.extract(case["text"])
        e2e = policy.decide(case["text"], candidate_count=len(facts))
        policy_only = policy.decide(case["text"], candidate_count=1)
        expected = (case["expected_action"], case["expected_reason"])
        details.append({
            **case,
            "extracted": [{"slot": f.slot, "value": f.value} for f in facts],
            "e2e_action": e2e.action,
            "e2e_reason": e2e.reason_code,
            "e2e_correct": (e2e.action, e2e.reason_code) == expected,
            "measurement_layer": "extractor_and_policy_only",
            "policy_action": policy_only.action,
            "policy_reason": policy_only.reason_code,
            "policy_correct": (policy_only.action, policy_only.reason_code) == expected,
        })

    summary = {}
    for mode in ("e2e", "policy"):
        for lang in ("zh", "en"):
            for split in (None, "dev", "heldout"):
                rows = [r for r in details if r["lang"] == lang and (split is None or r["split"] == split)]
                opt_out = [r for r in rows if r["kind"] == "opt_out"]
                explicit = [r for r in rows if r["kind"] == "explicit"]
                key = f"{mode}.{lang}" if split is None else f"{mode}.{lang}.{split}"
                summary[key] = {
                    "n": len(rows), "opt_out_n": len(opt_out), "explicit_n": len(explicit),
                    "accuracy": ratio(sum(r[f"{mode}_correct"] for r in rows), len(rows)),
                    "commit_violation_rate": ratio(sum(r[f"{mode}_action"] == "commit" for r in opt_out), len(opt_out)),
                    "propose_violation_rate": ratio(sum(r[f"{mode}_action"] == "propose" for r in opt_out), len(opt_out)),
                    # 用户明确拒绝保存，系统却写入或生成候选：最严重的错误
                    "violation_rate": ratio(
                        sum(r[f"{mode}_action"] in ("commit", "propose") for r in opt_out), len(opt_out)
                    ),
                    "missed_explicit_rate": ratio(
                        sum(r[f"{mode}_action"] != "commit" for r in explicit), len(explicit)
                    ),
                    "no_fact_rate": ratio(
                        sum(r[f"{mode}_reason"] == "no_extractable_fact" for r in rows), len(rows)
                    ),
                }
    return summary, {"consent": details}


# ---------------------------------------------------------------------------
# 3. 更新与删除（记忆生命周期）
# ---------------------------------------------------------------------------

class MemoryStack:
    """一套独立的 SQLite + FAISS 记忆存储，每个场景用一个临时目录。"""

    def __init__(self, root: Path, embedder, config, workspace_id: str = "local"):
        from memory.memory_manager import MemoryManager
        from product.memory_service import ProductMemoryService
        from product_store import ProductStore
        from retrieval.vector_store import VectorStore

        self.root = Path(root)
        self.db_path = self.root / "product.db"
        self.memory_dir = self.root / "memory"
        self.store = ProductStore(self.db_path, default_workspace_id=workspace_id)
        base_vs = VectorStore(str(self.root / "base"), config.VECTOR_DIM)
        memory_vs = VectorStore(str(self.memory_dir), config.VECTOR_DIM)
        self.mm = MemoryManager(
            base_vs=base_vs,
            memory_vs=memory_vs,
            embed_model=embedder,
            base_max_dist=config.BASE_MAX_DISTANCE,
            memory_max_dist=config.MEMORY_MAX_DISTANCE,
            base_recall_k=config.BASE_RECALL_K,
            memory_recall_k=config.MEMORY_RECALL_K,
            short_term_limit=config.SHORT_TERM_LIMIT,
            workspace_id=workspace_id,
        )
        self.svc = ProductMemoryService(self.store, self.mm, workspace_id)
        self.svc.initialize_index()

    def close(self):
        self.store.close()

    # ---- 读取工具 ----
    def active_vectors(self, memory_id=None):
        records = self.mm.list_active_memory_records(limit=None)
        if memory_id is None:
            return records
        return [r for r in records if r.get("memory_id") == memory_id]

    def all_vectors(self, memory_id):
        return self.mm.memory_vs.list_records({"kind": "user_memory", "memory_id": memory_id})

    def slot_lookup(self, slot):
        # 直接按槽位读取（绕过问题路由），只检查存储状态本身
        return self.mm._lookup_active_memory_slots([slot])

    def rendered(self, slot, value):
        return self.mm._render_memory_value(slot, value)

    def retrieve_all_values(self, query):
        """用产品自己的检索函数，但取消 top-k 和距离阈值：检查过滤逻辑，不依赖嵌入质量。"""
        saved = self.mm.memory_recall_k
        self.mm.memory_recall_k = max(1, self.mm.memory_vs.total_count)
        try:
            records = self.mm.retrieve_memory_records(query, max_distance=None)
        finally:
            self.mm.memory_recall_k = saved
        expected_ids = {item["memory_id"] for item in self.svc.list(limit=1_000_000)}
        returned_ids = {r["metadata"].get("memory_id") for r in records}
        if returned_ids != expected_ids:
            raise AssertionError("unbounded retrieval did not return exactly the active SQLite IDs")
        return [r["metadata"].get("value") for r in records]

    def topk_values(self, query):
        records = self.mm.retrieve_memory_records(query, max_distance=self.mm.memory_max_dist)
        return [r["metadata"].get("value") for r in records]

    def consistent(self):
        """SQLite 中的有效记忆与向量库中的有效记录一一对应，且值相同。"""
        sql = {m["memory_id"]: norm_value(m["value"]) for m in self.svc.list(limit=1_000_000)}
        vectors = self.active_vectors()
        vec = defaultdict(list)
        for record in vectors:
            vec[record.get("memory_id")].append(norm_value(record.get("value", "")))
        return set(sql) == set(vec) and all(vec[mid] == [value] for mid, value in sql.items())


def _contains_bytes(path: Path, value: str) -> bool:
    if not path.exists():
        return False
    data = path.read_bytes()
    needles = {value.encode("utf-8"), json.dumps(value).strip('"').encode("ascii")}
    return any(needle in data for needle in needles)


def _not_found(callable_):
    from product_store import NotFoundError
    try:
        callable_()
    except NotFoundError:
        return True
    return False


def run_lifecycle_scenario(scenario: dict, root: Path, models: Models) -> dict:
    embedder, config = models.embedder, models.config
    slot, old, new = scenario["slot"], scenario["old"], scenario["new"]
    kind = scenario["type"]
    checks = {}
    stack = MemoryStack(root, embedder, config)
    try:
        for item in scenario["distractors"]:
            stack.svc.remember(item["slot"], item["value"])

        def create_and_update():
            created = stack.svc.remember(slot, old)
            memory_id = created.memory["memory_id"]
            if scenario.get("update_path") == "remember":
                stack.svc.remember(slot, new)
            else:
                stack.svc.edit(memory_id, new)
            return memory_id

        if kind == "update":
            mid = create_and_update()
            current = stack.store.get_memory(mid)
            checks["sql_state"] = norm_value(current["value"]) == norm_value(new) and current["version"] == 2
            checks["vector_consistent"] = stack.consistent()
            history = [norm_value(h["value"]) for h in stack.svc.history(mid)]
            checks["history_correct"] = history == [norm_value(old), norm_value(new)]
            checks["slot_lookup_correct"] = stack.slot_lookup(slot) == [stack.rendered(slot, new)]
            checks["no_stale_leak"] = old not in stack.retrieve_all_values(scenario["semantic_query"])
            checks["semantic_hit"] = new in stack.topk_values(scenario["semantic_query"])

        elif kind == "delete":
            mid = stack.svc.remember(slot, old).memory["memory_id"]
            stack.svc.delete(mid)
            deleted = stack.store.get_memory(mid, include_deleted=True)
            checks["sql_state"] = (
                _not_found(lambda: stack.store.get_memory(mid))
                and deleted["status"] == "deleted"
                and all(m["memory_id"] != mid for m in stack.svc.list(limit=1_000_000))
            )
            checks["vector_consistent"] = stack.consistent() and not stack.active_vectors(mid)
            checks["slot_lookup_correct"] = stack.slot_lookup(slot) == []
            checks["no_stale_leak"] = old not in stack.retrieve_all_values(scenario["semantic_query"])

        elif kind == "update_delete":
            mid = create_and_update()
            stack.svc.delete(mid)
            values = stack.retrieve_all_values(scenario["semantic_query"])
            checks["sql_state"] = _not_found(lambda: stack.store.get_memory(mid))
            checks["vector_consistent"] = stack.consistent() and not stack.active_vectors(mid)
            checks["slot_lookup_correct"] = stack.slot_lookup(slot) == []
            checks["no_stale_leak"] = old not in values and new not in values

        elif kind == "purge":
            # 先更新再删除再清除：所有历史版本都必须消失
            mid = create_and_update()
            stack.svc.delete(mid)
            stack.svc.purge(mid)
            checks["purge_sql_gone"] = (
                _not_found(lambda: stack.store.get_memory(mid, include_deleted=True))
                and _not_found(lambda: stack.svc.history(mid))
            )
            texts = " ".join(r["text"] for r in stack.mm.memory_vs.list_records({"kind": "user_memory"}))
            checks["purge_vector_gone"] = not stack.all_vectors(mid) and old not in texts and new not in texts
            exported = json.dumps(stack.svc.export(), ensure_ascii=False)
            checks["purge_export_clean"] = old not in exported and new not in exported
            checks["secure_delete_setting"] = int(stack.store.connection.execute("PRAGMA secure_delete").fetchone()[0])
            db_files = [stack.db_path, Path(str(stack.db_path) + "-journal"), Path(str(stack.db_path) + "-wal")]
            checks["purge_db_bytes_clean"] = not any(
                _contains_bytes(path, value) for path in db_files for value in (old, new)
            )
            snapshot = stack.memory_dir / "store.snapshot"
            checks["purge_snapshot_bytes_clean"] = not any(_contains_bytes(snapshot, v) for v in (old, new))
            values = stack.retrieve_all_values(scenario["semantic_query"])
            checks["no_stale_leak"] = old not in values and new not in values

        elif kind in ("restart", "index_loss"):
            mid = create_and_update()
            stack.close()
            if kind == "index_loss":
                shutil.rmtree(_within(stack.memory_dir, root))
            stack = MemoryStack(root, embedder, config)
            current = stack.store.get_memory(mid)
            ok = (
                norm_value(current["value"]) == norm_value(new)
                and stack.consistent()
                and stack.slot_lookup(slot) == [stack.rendered(slot, new)]
            )
            checks["recovered_after_restart" if kind == "restart" else "recovered_after_index_loss"] = ok
            checks["no_stale_leak"] = old not in stack.retrieve_all_values(scenario["semantic_query"])
            checks["semantic_hit"] = new in stack.topk_values(scenario["semantic_query"])

        elif kind in ("candidate_confirm", "candidate_reject"):
            candidates = stack.svc.propose_extracted(scenario["propose_text"])
            created = (
                len(candidates) == 1
                and candidates[0]["slot"] == slot
                and norm_value(candidates[0]["value"]) == norm_value(new)
            )
            checks["candidate_created"] = created
            if created:
                cid = candidates[0]["candidate_id"]
                hidden = (
                    new not in stack.retrieve_all_values(scenario["semantic_query"])
                    and stack.slot_lookup(slot) == []
                    and all(m["slot"] != slot for m in stack.svc.list(limit=1_000_000))
                )
                if kind == "candidate_confirm":
                    checks["candidate_hidden"] = hidden
                    stack.svc.confirm_candidate(cid)
                    checks["candidate_confirmed_visible"] = (
                        any(m["slot"] == slot and norm_value(m["value"]) == norm_value(new)
                            for m in stack.svc.list(limit=1_000_000))
                        and stack.slot_lookup(slot) == [stack.rendered(slot, new)]
                        and stack.consistent()
                    )
                    checks["semantic_hit"] = new in stack.topk_values(scenario["semantic_query"])
                else:
                    stack.svc.reject_candidate(cid)
                    checks["candidate_rejected_hidden"] = (
                        new not in stack.retrieve_all_values(scenario["semantic_query"])
                        and stack.slot_lookup(slot) == []
                        and all(m["slot"] != slot for m in stack.svc.list(limit=1_000_000))
                    )
            else:
                # 前置条件失败：后续检查记为“不适用”
                if kind == "candidate_confirm":
                    checks.update({"candidate_hidden": None, "candidate_confirmed_visible": None, "semantic_hit": None})
                else:
                    checks["candidate_rejected_hidden"] = None
        else:
            raise ValueError(f"unknown lifecycle type: {kind}")
    finally:
        stack.close()
    return checks


SCENARIO_CHECKS = {
    "update": ("sql_state", "vector_consistent", "history_correct", "slot_lookup_correct", "no_stale_leak", "semantic_hit"),
    "delete": ("sql_state", "vector_consistent", "slot_lookup_correct", "no_stale_leak"),
    "update_delete": ("sql_state", "vector_consistent", "slot_lookup_correct", "no_stale_leak"),
    "purge": ("purge_sql_gone", "purge_vector_gone", "purge_export_clean", "purge_db_bytes_clean",
              "purge_snapshot_bytes_clean", "no_stale_leak"),
    "restart": ("recovered_after_restart", "no_stale_leak", "semantic_hit"),
    "index_loss": ("recovered_after_index_loss", "no_stale_leak", "semantic_hit"),
    "candidate_confirm": ("candidate_created", "candidate_hidden", "candidate_confirmed_visible", "semantic_hit"),
    "candidate_reject": ("candidate_created", "candidate_rejected_hidden"),
}

def run_lifecycle(cases_dir: Path, models: Models):
    details = []
    with tempfile.TemporaryDirectory(prefix="dm_eval_", dir=TEMP_ROOT) as tmp:
        for scenario in load_jsonl(cases_dir / "lifecycle.jsonl"):
            root = _within(Path(tmp) / _component(scenario["id"]), Path(tmp))
            try:
                checks = run_lifecycle_scenario(scenario, root, models)
                error = None
            except Exception as exc:  # 场景报错本身就是一个失败结果
                checks = {key: False for key in SCENARIO_CHECKS.get(scenario["type"], ())}
                error = f"{type(exc).__name__}: {exc}"
            details.append({**scenario, "checks": checks, "error": error})

    summary = {"by_check": {}, "errors": sum(1 for d in details if d["error"]),
               "secure_delete_observed": sorted({d["checks"]["secure_delete_setting"] for d in details
                                                  if "secure_delete_setting" in d["checks"]})}
    for name, _label, _semantic in LIFECYCLE_CHECKS:
        for lang in ("zh", "en"):
            rows = [d for d in details if d["lang"] == lang and name in d["checks"]]
            applicable = [d for d in rows if d["checks"][name] is not None]
            if not rows:
                continue
            summary["by_check"][f"{name}.{lang}"] = {
                "n": len(applicable),
                "not_applicable": len(rows) - len(applicable),
                "pass_rate": ratio(sum(1 for d in applicable if d["checks"][name]), len(applicable)),
            }
    for lang in ("zh", "en"):
        rows = [d for d in details if d["lang"] == lang]
        summary["by_check"][f"scenario_errors.{lang}"] = {
            "n": len(rows), "not_applicable": 0,
            "pass_rate": ratio(sum(1 for d in rows if not d["error"]), len(rows)),
        }
    return summary, {"lifecycle": details}


# ---------------------------------------------------------------------------
# 4. 端到端延迟
# ---------------------------------------------------------------------------

CAREER_NOTES = """# 求职准备笔记（虚构示例）

准备机器学习岗位面试时，先复习监督学习、评估指标和常见模型的优缺点。
项目经历要讲清楚问题、方法、结果和自己的贡献。

# Job search notes (fictional example)

Tech companies often look for solid programming skills, clear communication and evidence of shipped projects.
RAG combines retrieval over a document collection with text generation by a language model.
"""


def run_latency(cases_dir: Path, models: Models, repeats: int):
    from memory.memory_manager import MemoryManager
    from pipeline.rag_pipeline import RAGPipeline
    from product.runtime import build_product_runtime
    from retrieval.vector_store import VectorStore

    config = models.config
    messages = load_jsonl(cases_dir / "latency_messages.jsonl")
    details = []
    # 每次重复均新建运行环境和会话，并使用相同的测试数据序列。
    with tempfile.TemporaryDirectory(prefix="dm_latency_", dir=TEMP_ROOT) as directory:
        root = Path(directory)
        for repeat in range(repeats):
            tmp = root / f"repeat-{repeat}"
            def engine_factory():
                base_vs = VectorStore(str(tmp / "base"), config.VECTOR_DIM)
                memory_vs = VectorStore(str(tmp / "memory"), config.VECTOR_DIM)
                manager = MemoryManager(
                    base_vs=base_vs, memory_vs=memory_vs, embed_model=models.embedder,
                    base_max_dist=config.BASE_MAX_DISTANCE, memory_max_dist=config.MEMORY_MAX_DISTANCE,
                    base_recall_k=config.BASE_RECALL_K, memory_recall_k=config.MEMORY_RECALL_K,
                    short_term_limit=config.SHORT_TERM_LIMIT,
                )
                return RAGPipeline(models.chat_llm(), manager), memory_vs
            runtime = build_product_runtime(
                engine_factory=engine_factory, db_path=tmp / "product.db",
                documents_path=tmp / "documents",
            )
            try:
                app = runtime.application
                notes = tmp / "career_notes.md"
                notes.write_text(CAREER_NOTES, encoding="utf-8")
                app.import_document(notes, source_id="career-notes", title="Career notes")
                session_id = app.create_session("latency")["session_id"]
                for message in messages:
                    started = time.perf_counter()
                    error = None
                    try:
                        result = app.chat(session_id, message["text"])
                        route = (result.get("user_message", {}).get("metadata") or {}).get("intent") \
                            or result.get("trace", {}).get("intent", "unknown")
                        trace = result.get("trace", {})
                        if trace.get("status") in {"partial", "failed"}:
                            error = trace.get("answer_error") or trace["status"]
                    except Exception as exc:
                        route, error = "error", f"{type(exc).__name__}: {exc}"
                    elapsed = (time.perf_counter() - started) * 1000
                    details.append({"id": message["id"], "lang": message["lang"], "text": message["text"],
                                    "repeat": repeat, "route": route, "latency_ms": elapsed, "error": error})
                    print(f"  延迟 {route:22s} {elapsed / 1000:.2f}s", flush=True)
            finally:
                runtime.close()
    summary = {}
    for route in sorted({d["route"] for d in details if d["route"] != "error"}) + ["all"]:
        rows = [d for d in details if route == "all" or d["route"] == route]
        successes = [d["latency_ms"] for d in rows if d["error"] is None]
        summary[route] = {"n": len(rows), "successful_n": len(successes),
                          "error_n": len(rows) - len(successes),
                          "p50": percentile(successes, 50), "p95": percentile(successes, 95)}
    return summary, {"latency": details}


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------

def headline(summary: dict, models: Models) -> dict:
    """抽出最重要的指标，用于前后对比。"""
    h = {}
    ext = summary.get("extraction")
    if ext:
        for key in ("zh", "en", "mix", "zh.heldout", "en.heldout"):
            if key in ext["groups"]:
                h[f"抽取F1·{key}"] = ext["groups"][key]["f1"]
        for lang in ("zh", "en"):
            h[f"负例误抽率·{lang}"] = ext["negative"][lang]["fp_rate"]
            h[f"召回槽位识别准确率·{lang}"] = ext["routing"][lang]["accuracy"]
    con = summary.get("consent")
    if con:
        for lang in ("zh", "en"):
            h[f"同意判定准确率·抽取+策略·{lang}"] = con[f"e2e.{lang}"]["accuracy"]
            h[f"拒绝保存违规决策率·抽取+策略·{lang}"] = con[f"e2e.{lang}"]["violation_rate"]
            h[f"拒绝保存时预测commit·{lang}"] = con[f"e2e.{lang}"]["commit_violation_rate"]
            h[f"拒绝保存时预测propose·{lang}"] = con[f"e2e.{lang}"]["propose_violation_rate"]
            h[f"同意判定准确率·仅策略·{lang}"] = con[f"policy.{lang}"]["accuracy"]
    life = summary.get("lifecycle")
    if life:
        for name, label, semantic in LIFECYCLE_CHECKS:
            if semantic and models.embedder_kind != "real":
                continue
            for lang in ("zh", "en"):
                item = life["by_check"].get(f"{name}.{lang}")
                if item:
                    h[f"{label}·{lang}"] = item["pass_rate"]
    lat = summary.get("latency")
    if lat and models.llm_kind == "real":
        h["单轮延迟中位数(ms)"] = lat["all"]["p50"]
        h["单轮延迟P95(ms)"] = lat["all"]["p95"]
    return h


def failure_samples(details: dict, limit: int = 5, skip_semantic: bool = False) -> list:
    lines = []
    ext = details.get("extraction")
    if ext:
        lines.append("### 抽取错误样例")
        for lang in ("zh", "en", "mix"):
            fails, seen_templates = [], set()
            for d in ext:  # 每种句式只举一个例子，避免样例全是同一个槽位
                if d["lang"] == lang and not d["correct"] and d["template"] not in seen_templates:
                    seen_templates.add(d["template"])
                    fails.append(d)
            for d in fails[:limit]:
                lines.append(f"- [{LANG_LABEL[lang]}·{d['kind']}] `{d['text']}` → 期望 {d['expected']}，实际 {d['predicted']}")
    con = details.get("consent")
    if con:
        lines.append("\n### 同意判定错误样例（抽取+策略联测）")
        for lang in ("zh", "en"):
            fails, seen_wrappers = [], set()
            for d in con:  # 每种包装句式只举一个例子
                key = d.get("wrapper", d["text"])
                if d["lang"] == lang and not d["e2e_correct"] and key not in seen_wrappers:
                    seen_wrappers.add(key)
                    fails.append(d)
            for d in fails[:limit]:
                lines.append(
                    f"- [{LANG_LABEL[lang]}·{d['kind']}] `{d['text']}` → 期望 {d['expected_action']}/{d['expected_reason']}，"
                    f"实际 {d['e2e_action']}/{d['e2e_reason']}"
                )
    life = details.get("lifecycle")
    if life:
        lines.append("\n### 生命周期失败样例")
        shown = 0
        semantic = {name for name, _label, is_semantic in LIFECYCLE_CHECKS if is_semantic}
        for d in life:
            failed = [name for name, value in d["checks"].items()
                      if value is False and not (skip_semantic and name in semantic)]
            if d["error"] or failed:
                lines.append(f"- {d['id']} [{LANG_LABEL[d['lang']]}·{d['type']}·{d['slot']}] "
                             f"失败项：{', '.join(failed) or '-'} {('报错：' + d['error']) if d['error'] else ''}")
                shown += 1
            if shown >= limit * 2:
                break
    return lines


def render_report(summary: dict, details: dict, meta: dict, models: Models) -> str:
    out = ["# DiscreteMemory 评测报告", ""]
    out.append(md_table(["项目", "值"], [
        ["运行名称", meta["name"]], ["时间", meta["time"]], ["代码版本", meta["git"]],
        ["嵌入模型", meta["models"]["embedder"]], ["LLM", meta["models"]["llm"]],
        ["用例版本", f"{meta['cases']['cases_version']}（seed={meta['cases']['seed']}，指纹 {meta['cases_fingerprint']}）"],
        ["Python", meta["python"]], ["SQLite", meta["sqlite"]],
        ["评测协议", meta["implementation"]["protocol"]],
        ["运行代码 SHA256", meta["implementation"]["runtime_sha256"][:12]],
        ["评测代码 SHA256", meta["implementation"]["evaluator_sha256"][:12]],
        ["执行参数", json.dumps(meta["run_options"], ensure_ascii=False)],
        ["输出目录", meta["output_dir"]],
    ]))
    if models.embedder_kind == "hash" or models.llm_kind == "stub":
        out += ["", "> ⚠️ 本次使用了替身模型。依赖模型的指标（语义检索命中、延迟）"
                "只用于验证评测脚本，不代表系统表现，已在表中标出或省略。"]

    out += ["", "> 固定用例由公开模板生成。heldout 是公开的模板诊断保留集，不能当作未见过的盲测集或真实用户分布。"]
    ext = summary.get("extraction")
    if ext:
        out += ["", "## 1. 记忆抽取", "",
                "按“槽位 + 归一化值”一致计分（忽略大小写、空白及首尾标点）；“槽位F1”只看槽位。heldout 使用不同的公开句式。", ""]
        rows = []
        for key, label in [("zh", "中文·全部"), ("zh.dev", "中文·dev"), ("zh.heldout", "中文·heldout"),
                           ("zh.compound", "中文·一句多事实"), ("en", "英文·全部"), ("en.dev", "英文·dev"),
                           ("en.heldout", "英文·heldout"), ("en.compound", "英文·一句多事实"), ("mix", "中英混合")]:
            g = ext["groups"].get(key)
            if g:
                rows.append([label, g["n"], pct(g["precision"]), pct(g["recall"]), pct(g["f1"]), pct(g["slot_f1"])])
        out.append(md_table(["子集", "用例数", "精确率", "召回率", "F1", "槽位F1"], rows))
        out += ["", md_table(["语言", "负例数", "负例误抽率", "误抽类别"], [
            [LANG_LABEL[lang], ext["negative"][lang]["n"], pct(ext["negative"][lang]["fp_rate"]),
             ", ".join(f"{k}×{v}" for k, v in ext["negative"][lang]["by_category"].items()) or "—"]
            for lang in ("zh", "en")
        ])]
        out += ["", "**召回槽位识别**（只测 infer_recall_slots，不包含完整问答路由、检索和回答）", ""]
        out.append(md_table(["语言", "dev", "heldout", "全部"], [
            [LANG_LABEL[lang], pct(ext["routing"][f"{lang}.dev"]["accuracy"]),
             pct(ext["routing"][f"{lang}.heldout"]["accuracy"]), pct(ext["routing"][lang]["accuracy"])]
            for lang in ("zh", "en")
        ]))

    con = summary.get("consent")
    if con:
        out += ["", "## 2. 同意判定", "",
                "抽取+策略 = 调用抽取器后调用写入策略；仅策略 = 固定 candidate_count=1，只测写入策略。"
                "两者均为决策层评测，不执行 ProductApplication.chat、候选持久化或长期写入。准确率要求 action 和 reason 同时匹配。"
                "拒绝保存违规决策率 = 拒绝例中预测为 commit 或 propose 的比例；commit 与 propose 分开列出，候选不等于 active 长期记忆。", ""]
        rows = []
        for mode, mode_label in (("e2e", "抽取+策略"), ("policy", "仅策略")):
            for lang in ("zh", "en"):
                s = con[f"{mode}.{lang}"]
                rows.append([f"{LANG_LABEL[lang]}·{mode_label}", s["n"], pct(s["accuracy"]),
                             pct(con[f"{mode}.{lang}.heldout"]["accuracy"]),
                                                           s["opt_out_n"], pct(s["violation_rate"]),
                              pct(s["commit_violation_rate"]), pct(s["propose_violation_rate"]),
                              pct(s["missed_explicit_rate"]), pct(s["no_fact_rate"])])
        out.append(md_table(["语言·模式", "用例数", "判定准确率", "heldout 准确率", "拒绝例数",
                             "违规决策率", "预测commit率", "预测propose率",
                             "明确保存漏判率", "无可抽取事实率"], rows))

    life = summary.get("lifecycle")
    if life:
        out += ["", "## 3. 更新与删除", "",
                "每个场景使用独立的临时 SQLite + FAISS。“全量检索”取消 top-k 和距离阈值，"
                "只检查过滤逻辑，不依赖嵌入质量。", ""]
        rows = []
        for name, label, semantic in LIFECYCLE_CHECKS:
            cells = []
            present = False
            for lang in ("zh", "en"):
                item = life["by_check"].get(f"{name}.{lang}")
                if not item:
                    cells.append("—")
                    continue
                present = True
                if semantic and models.embedder_kind != "real":
                    cells.append("需真实嵌入模型")
                elif item["n"] == 0:
                    cells.append(f"不适用（前置步骤失败 {item['not_applicable']} 个）")
                else:
                    na = f"，{item['not_applicable']} 个不适用" if item["not_applicable"] else ""
                    cells.append(f"{pct(item['pass_rate'])}（n={item['n']}{na}）")
            if present:
                rows.append([label, *cells, "是" if semantic else "否"])
        out.append(md_table(["检查项", "中文", "英文", "依赖嵌入模型"], rows))
        out += ["", f"本次临时产品数据库实际 secure_delete 值：{life['secure_delete_observed']}。"
                "字节检查只扫描该场景的指定 SQLite/journal/WAL 与当前向量快照，不能证明备份、聊天消息、知识文档或存储介质被彻底擦除。",
                "英文直接 slot/value 状态操作与英文自然语言候选创建分开统计；前置失败标为不适用，运行异常计入失败。"]
        if life["errors"]:
            out += ["", f"有 {life['errors']} 个场景运行时报错，见失败样例。"]

    lat = summary.get("latency")
    if lat:
        out += ["", "## 4. 单轮对话延迟", ""]
        if models.llm_kind != "real":
            out.append("> 替身模式：只反映检索与存储的开销，不含真实模型推理时间。\n")
        out.append("每轮重新创建隔离运行时与会话；计时为 ProductApplication.chat 调用，排除启动/模型加载。"
                   "同一轮内保留固定消息顺序，模型设备保持已加载状态；成功请求计分，错误数另列。样本很少，P95 仅描述本机样本。")
        out.append(md_table(["路由", "请求数", "成功数", "错误数", "成功请求中位数", "成功请求P95"], [
            [route, s["n"], s["successful_n"], s["error_n"], ms(s["p50"]), ms(s["p95"])] for route, s in lat.items()
        ]))

    out += ["", "## 失败样例", ""] + failure_samples(details, skip_semantic=(models.embedder_kind == "hash"))
    return "\n".join(out) + "\n"


def git_version() -> str:
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT_ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
        return commit + ("（有未提交的修改）" if dirty else "")
    except Exception:
        return "未知"


def sqlite_info() -> str:
    import sqlite3
    try:
        with sqlite3.connect(":memory:") as connection:
            secure = connection.execute("PRAGMA secure_delete").fetchone()[0]
    except Exception:
        secure = "?"
    return f"{sqlite3.sqlite_version}，secure_delete 默认={secure}"


def cases_fingerprint(cases_dir: Path) -> str:
    return source_fingerprint(list(cases_dir.glob("*.jsonl")) + [cases_dir / "manifest.json"], cases_dir)


def compare(dir_a: Path, dir_b: Path) -> str:
    a = json.loads((dir_a / "summary.json").read_text(encoding="utf-8"))
    b = json.loads((dir_b / "summary.json").read_text(encoding="utf-8"))
    incompatible = []
    for key in ("cases_fingerprint", "embedder_kind", "llm_kind", "models", "suites", "run_options"):
        if a["meta"].get(key) != b["meta"].get(key):
            incompatible.append(key)
    for key in ("protocol", "runtime_config"):
        if a["meta"].get("implementation", {}).get(key) != b["meta"].get("implementation", {}).get(key):
            incompatible.append(key)
    if a["meta"].get("executed_case_ids") != b["meta"].get("executed_case_ids"):
        incompatible.append("executed_case_ids")
    if incompatible:
        raise ValueError("不可直接比较：评测数据、口径、模型或执行范围不同：" + ", ".join(incompatible))
    keys = list(dict.fromkeys(list(a["headline"]) + list(b["headline"])))
    rows = []
    for key in keys:
        va, vb = a["headline"].get(key), b["headline"].get(key)
        is_ms = "(ms)" in key
        fmt = ms if is_ms else pct
        delta = "—" if va is None or vb is None else (
            f"{vb - va:+.0f} ms" if is_ms else f"{(vb - va) * 100:+.1f} 个百分点")
        rows.append([key, fmt(va), fmt(vb), delta])
    title = f"## 前后对比：{a['meta']['name']} → {b['meta']['name']}\n\n"
    return title + md_table(["指标", a["meta"]["name"], b["meta"]["name"], "变化"], rows) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="DiscreteMemory 离线评测")
    parser.add_argument("--suites", default=",".join(ALL_SUITES),
                        help="逗号分隔：extraction,consent,lifecycle,latency")
    parser.add_argument("--embedder", choices=["real", "hash"], default="real")
    parser.add_argument("--llm", choices=["real", "stub", "none"], default="real")
    parser.add_argument("--name", default=None,
                        help="自定义结果目录名；默认按评测内容、运行模式和时间命名")
    parser.add_argument("--cases", type=Path, default=EVAL_DIR / "cases")
    parser.add_argument("--out", type=Path, default=RESULTS_ROOT,
                        help="结果根目录；默认项目 eval/results")
    parser.add_argument("--latency-repeats", type=int, default=3)
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("A", "B"))
    args = parser.parse_args()

    if args.compare:
        try:
            table = compare(*args.compare)
        except (OSError, KeyError, ValueError) as exc:
            parser.error(str(exc))
        print(table)
        target = args.compare[1] / f"compare_vs_{args.compare[0].name}.md"
        target.write_text(table, encoding="utf-8")
        print(f"已写入 {target}")
        return

    suites = [s.strip() for s in args.suites.split(",") if s.strip()]
    unknown = set(suites) - set(ALL_SUITES)
    if unknown:
        parser.error(f"未知的评测类别：{', '.join(sorted(unknown))}")
    if not suites:
        parser.error("至少指定一个评测类别")
    if len(suites) != len(set(suites)):
        parser.error("评测类别不能重复")
    if args.latency_repeats <= 0:
        parser.error("--latency-repeats 必须是正整数")
    if args.llm == "none":
        skipped = [s for s in suites if s in ("latency",)]
        if skipped:
            print(f"[提示] --llm none，跳过：{', '.join(skipped)}")
        suites = [s for s in suites if s not in ("latency",)]
    if not suites:
        parser.error("当前模型选项下没有可运行的评测类别")
    needs_embedder = any(s in ("lifecycle", "latency") for s in suites)
    models = Models(args.embedder if needs_embedder else "hash",
                    args.llm if any(s in ("latency",) for s in suites) else "none")

    name = args.name or _default_run_name(
        suites, args.embedder, args.llm,
        latency_repeats=args.latency_repeats,
    )
    try:
        out_dir = _create_result_directory(
            _local_path(args.out).resolve(), name, automatic=not bool(args.name),
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    name = out_dir.name
    if needs_embedder:
        TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.cases / "manifest.json").read_text(encoding="utf-8"))

    summary, details, timings = {}, {}, {}
    runners = {
        "extraction": lambda: run_extraction(args.cases),
        "consent": lambda: run_consent(args.cases),
        "lifecycle": lambda: run_lifecycle(args.cases, models),
        "latency": lambda: run_latency(args.cases, models, args.latency_repeats),
    }
    for suite in suites:
        print(f"== {suite}", flush=True)
        started = time.perf_counter()
        suite_summary, suite_details = runners[suite]()
        timings[suite] = round(time.perf_counter() - started, 2)
        summary[suite] = suite_summary
        details.update(suite_details)

    if not needs_embedder:
        models.embedder_kind = "none"
    meta = {
        "name": name,
        "time": _now().isoformat(timespec="seconds"),
        "git": git_version(),
        "models": models.describe() if needs_embedder or models.llm_kind != "none" else
        {"embedder": "不需要", "llm": "不需要"},
        "embedder_kind": models.embedder_kind if needs_embedder else None,
        "llm_kind": models.llm_kind,
        "suites": suites,
        "suite_seconds": timings,
        "cases": manifest,
        "cases_fingerprint": cases_fingerprint(args.cases),
        "python": f"{platform.python_version()} ({platform.system()})",
        "sqlite": sqlite_info(),
        "implementation": implementation_metadata(models.config),
        "run_options": {"latency_repeats": args.latency_repeats},
        "executed_case_ids": {key: [r["id"] for r in rows] for key, rows in details.items()},
        "output_dir": str(out_dir),
    }
    result = {"meta": meta, "headline": headline(summary, models), "summary": summary}
    (out_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    for name_, rows in details.items():
        write_jsonl(out_dir / "details" / f"{name_}.jsonl", rows)
    report = render_report(summary, details, meta, models)
    (out_dir / "report.md").write_text(report, encoding="utf-8")

    print("\n关键指标：")
    for key, value in result["headline"].items():
        print(f"  {key:36s} {ms(value) if '(ms)' in key else pct(value)}")
    print(f"\n报告：{out_dir / 'report.md'}")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    main()
