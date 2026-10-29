"""检索侧配对消融：对相同问题，比较提供与不提供检索记忆的结果。

python eval/memory_ablation.py
python eval/memory_ablation.py --device cpu
python eval/memory_ablation.py --summarize eval/results/<run>

正确的合成记忆通过应用接口写入，不经过大模型提取器。
两组均使用现有应用系统提示词调用真实聊天模型。
唯一的干预是移除 memory_context，并绕过直接返回固定回答的分支。
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from urllib.parse import urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from eval.run_eval import (
    RESULTS_ROOT, TEMP_ROOT, _create_result_directory, _local_path, _now,
    implementation_metadata, source_fingerprint,
)

PROTOCOL = "memory-read-ablation-1.0"
ARMS = ("with_memory", "without_memory")
LABELS = {"with_memory": "有长期记忆", "without_memory": "无长期记忆"}
KINDS = {"memory": "长期事实问答", "boundary": "无有效信息时克制", "history": "近期对话控制", "general": "通用知识控制"}
UNKNOWN_WORDS = ("不知道", "不清楚", "没有", "未提供", "未告诉", "无法确定", "无法得知", "不确定", "未知", "不知晓", "不掌握", "不了解", "尚未")


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def answer_hash(answer):
    return hashlib.sha256(answer.encode("utf-8")).hexdigest()


def load_cases(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    ids = [case["id"] for case in data["cases"]]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("cases must have nonempty, unique IDs")
    for case in data["cases"]:
        if case["kind"] not in KINDS or not case["question"].strip() or not case["reference"].strip():
            raise ValueError(f"invalid case: {case['id']}")
        if case["kind"] != "boundary" and not case.get("required"):
            raise ValueError("answerable cases need required alias groups")
        for fact in case.get("memories", []):
            if sum(bool(fact.get(key)) for key in ("pending", "delete", "update")) > 1:
                raise ValueError("one lifecycle operation per fact")
    return data


def seed_case(runtime, case):
    """使用受控测试数据验证真实应用的写入、更新和删除操作。"""
    from memory.extractor import ExtractedMemory
    from memory.model_extractor import ModelMemoryExtractor
    app = runtime.application
    source_session = app.create_session("合成事实来源")["session_id"]
    expected_active, expected_pending = [], []
    for fact in case.get("memories", []):
        slot, value = fact["slot"], fact["value"]
        source = f"评测构造的用户事实：{slot} = {value}"
        message = app.store.add_message(source_session, "user", source, workspace_id=app.workspace_id)
        if fact.get("pending"):
            app.memory.propose_extracted(
                source, session_id=source_session, source_message_id=message["message_id"],
                facts=[ExtractedMemory(ModelMemoryExtractor.SLOTS[slot][0], slot, value, source)],
            )
            expected_pending.append((slot, value))
            continue
        outcome = app.create_memory(
            slot, value, cardinality=fact.get("cardinality", "scalar"),
            session_id=source_session, source_message_id=message["message_id"],
            provenance={"evaluation": PROTOCOL, "fixture": case["id"]},
        )
        memory_id = outcome["memory"]["memory_id"]
        version = 1
        if "update" in fact:
            value, version = fact["update"], 2
            app.update_memory(memory_id, value)
        if fact.get("delete"):
            app.delete_memory(memory_id)
        else:
            expected_active.append((slot, value, version))
    active, pending = app.list_memories(limit=10_000), app.list_memory_candidates(limit=10_000)
    assert sorted((r["slot"], r["value"], r["version"]) for r in active) == sorted(expected_active)
    assert sorted((r["slot"], r["value"]) for r in pending) == sorted(expected_pending)
    projection = app.manager.list_active_memory_records(limit=10_000)
    assert {(r["memory_id"], r["slot"], r["value"]) for r in projection} == {
        (r["memory_id"], r["slot"], r["value"]) for r in active
    }
    # 新会话无法看到来源会话的消息。仅明确设置的对照用例
    # 会提供近期历史，且两组获得完全相同的历史。
    app.create_session("独立提问会话")
    app.manager.short_term_history = copy.deepcopy(case.get("history", []))
    return {"active": active, "pending": pending, "state_checks_passed": True}


def paired_messages(engine, case):
    decision = engine.query_policy.decide(case["question"])
    if decision.intent in {"memory_recall", "personal_question"}:
        mode = "personal"
        contexts = engine.mm.get_memory_recall_context(case["question"])
    elif decision.intent in {"normal_rag", "document_qa"}:
        mode = "document" if decision.intent == "document_qa" else "general"
        contexts = engine.mm.get_long_term_context(case["question"])
    else:
        raise ValueError(f"ablation requires a read-only question: {decision.intent}")
    masked = {**contexts, "memory_context": "无。", "memory_records": []}
    messages = {
        "with_memory": engine.build_messages(case["question"], contexts, mode, user_facts=case.get("current_facts")),
        "without_memory": engine.build_messages(case["question"], masked, mode, user_facts=case.get("current_facts")),
    }
    # 两组的对话内容和非记忆背景必须完全相同。
    assert messages["with_memory"][1:] == messages["without_memory"][1:]
    assert engine.mm.short_term_history == case.get("history", [])
    return messages, {
        "route": decision.intent, "answer_mode": mode, "contexts": contexts,
        "retrieval": copy.deepcopy(engine.mm.last_retrieval_trace),
    }


def keyword_check(case, answer, error=None):
    """仅作为初步筛查信号；否定、矛盾和编造内容仍需人工复核。"""
    normalized = "".join(answer.lower().split())
    groups_ok = all(any("".join(term.lower().split()) in normalized for term in group)
                    for group in case.get("required", []))
    forbidden = [term for term in case.get("forbidden", []) if term.lower() in answer.lower()]
    unknown = any(term in answer for term in UNKNOWN_WORDS)
    passed = bool(answer.strip()) and not error and groups_ok and not forbidden
    if case["kind"] == "boundary":
        passed = passed and unknown
    return {"passed": bool(passed), "required_groups_found": groups_ok,
            "forbidden_mentions": forbidden, "unknown_word_found": unknown,
            "stale_mentions": [term for term in case.get("stale_values", []) if term.lower() in answer.lower()]}


def summarize(run):
    run = Path(run)
    meta = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
    data = load_cases(run / "cases.json")
    cases = {c["id"]: c for c in data["cases"]}
    rows = [json.loads(line) for line in (run / "details.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    keys = {(row["case_id"], row["arm"]) for row in rows}
    if len(keys) != len(rows) or any(key[0] not in cases or key[1] not in ARMS for key in keys):
        raise ValueError("invalid or duplicate result rows")
    review_path = run / "review.json"
    review = json.loads(review_path.read_text(encoding="utf-8")) if review_path.exists() else None
    reviewed = {}
    if review:
        for item in review["items"]:
            key = (item["case_id"], item["arm"])
            if key in reviewed or key not in keys:
                raise ValueError(f"invalid review key: {key}")
            row = next(row for row in rows if (row["case_id"], row["arm"]) == key)
            if item["answer_sha256"] != row["answer_sha256"]:
                raise ValueError(f"review does not match this exact answer: {key}")
            if any(type(item[field]) is not bool for field in ("correct", "unsupported_personal_fact", "stale_fact")):
                raise ValueError("review labels must be booleans")
            if not item.get("reason") or (row.get("error") and item["correct"]):
                raise ValueError("review needs a reason and cannot pass an execution error")
            reviewed[key] = item
    complete = len(rows) == 2 * len(cases)
    review_complete = complete and len(reviewed) == len(rows)
    metrics = {}
    for kind, label in KINDS.items():
        ids = {c["id"] for c in cases.values() if c["kind"] == kind}
        metrics[kind] = {"label": label, "cases": len(ids)}
        for arm in ARMS:
            subset = [r for r in rows if r["case_id"] in ids and r["arm"] == arm]
            metrics[kind][arm] = {
                "completed": len(subset), "errors": sum(bool(r.get("error")) for r in subset),
                "keyword_passed": sum(r["keyword_check"]["passed"] for r in subset),
                "review_correct": sum(reviewed[(r["case_id"], arm)]["correct"] for r in subset) if review_complete else None,
            }
    audit = {}
    for arm in ARMS:
        personal = [r for r in rows if r["arm"] == arm and cases[r["case_id"]]["kind"] != "general"]
        updates = [r for r in personal if cases[r["case_id"]].get("stale_values")]
        audit[arm] = {
            "personal_cases": len(personal), "stale_applicable_cases": len(updates),
            "unsupported_personal_facts": sum(reviewed[(r["case_id"], arm)]["unsupported_personal_fact"] for r in personal) if review_complete else None,
            "stale_fact_errors": sum(reviewed[(r["case_id"], arm)]["stale_fact"] for r in updates) if review_complete else None,
        }
    summary = {"metadata": meta, "complete": complete, "review_complete": review_complete,
               "reviewer": review.get("reviewer") if review else None,
               "questions": len(cases), "responses": len(rows), "metrics": metrics, "audit": audit}
    dump(run / "summary.json", summary)
    lines = ["# 长期记忆有无对照", "", f"协议：{PROTOCOL}；时间：{meta['time']}；模型：{meta['model']}。",
             f"共 {len(cases)} 道题，已记录 {len(rows)} 个回答。设备：{meta['device']}。", "",
             "## 比较方法", "",
             "通过产品 API 构造已知事实，执行更新/删除/候选操作，并检查 SQLite 与 FAISS 一致。提问时丢弃来源会话历史；近期对话控制题除外。",
             "使用现有路由、BGE/FAISS 或槽位读取、系统提示词。两组均调用同一真实 LLM，只将无记忆组的长期记忆上下文置为空，不使用生产流程中的固定未知回答。",
             "这是给定正确记忆的读取侧对照，不测自然语言抽取、写入授权识别或完整端到端成功率。公开合成小样本、单模型单次运行，不是盲测。",
             "无记忆组对跨会话事实表示未知属于未完成事实回答，但不等于编造；无有效信息题和控制题单独统计。", "",
             "## 结果", "", "规则检查仅识别词语，不判断完整语义。复核必须逐条核对正确性、无依据的个人事实和过时事实。",
             "复核状态：" + (str(review['reviewer']) if review_complete else "尚未完成，不能将规则分数称为回答准确率。"), "",
             "| 题型 | 题数 | 有记忆：规则通过 | 无记忆：规则通过 | 有记忆：复核答对 | 无记忆：复核答对 |",
             "|---|---:|---:|---:|---:|---:|"]
    for m in metrics.values():
        n = m["cases"]
        values = [str(m[a]["keyword_passed"]) + f"/{n}" for a in ARMS]
        values += [str(m[a]["review_correct"]) + f"/{n}" if review_complete else "待复核" for a in ARMS]
        lines.append(f"| {m['label']} | {n} | " + " | ".join(values) + " |")
    if review_complete:
        lines.extend(["", "| 错误项 | 有记忆 | 无记忆 |", "|---|---:|---:|"])
        for label, numerator, denominator in (("无依据个人事实", "unsupported_personal_facts", "personal_cases"),
                                               ("旧事实作为当前值", "stale_fact_errors", "stale_applicable_cases")):
            lines.append(f"| {label} | " + " | ".join(f"{audit[a][numerator]}/{audit[a][denominator]}" for a in ARMS) + " |")
    lines.extend(["", "执行错误计入失败；相同问题的两组顺序交替。每组最多生成同样的 token 数，保留截断原因和耗时；本评测不比较性能延迟。", "", "## 逐题证据", ""])
    for case in cases.values():
        lines.extend([f"### {case['id']} · {case['category']}", "", "问题：" + case["question"], "", "判据：" + case["reference"], ""])
        for arm in ARMS:
            row = next((r for r in rows if r["case_id"] == case["id"] and r["arm"] == arm), None)
            if row is None:
                lines.extend([LABELS[arm] + "：未完成", ""])
                continue
            item = reviewed.get((case["id"], arm))
            lines.extend([f"**{LABELS[arm]}**", "", row["answer"] or row.get("error", "空回答"), ""])
            if item:
                lines.extend(["复核：" + ("正确。" if item["correct"] else "未答对。") + item["reason"], ""])
            else:
                lines.extend(["规则检查：" + str(row["keyword_check"]), ""])
    (run / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def run_evaluation(args):
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import requests
    import torch
    from config import Config
    from models.embedding import EmbeddingModel
    from models.llm import LLMModel
    from product.runtime import build_engine, build_product_runtime

    data = load_cases(args.cases)
    parts = urlsplit(Config.OLLAMA_URL)
    api = urlunsplit(parts._replace(path=parts.path.rsplit("/", 1)[0], query="", fragment=""))
    tags = requests.get(api + "/tags", timeout=10)
    tags.raise_for_status()
    model_info = next((item for item in tags.json()["models"] if item["name"] == Config.LLM_MODEL), None)
    if model_info is None:
        raise ValueError(f"Ollama has no configured model {Config.LLM_MODEL}; no model is downloaded automatically")
    version = requests.get(api + "/version", timeout=10)
    version.raise_for_status()
    torch.set_num_threads(4)
    options = {"seed": args.seed, "num_ctx": 4096, "num_predict": 256}
    if args.device == "cpu":
        options.update(num_gpu=0, num_thread=4)
    llm = LLMModel(Config.OLLAMA_URL, Config.LLM_MODEL, timeout=args.timeout, options=options)
    embedder = EmbeddingModel(Config.EMBED_MODEL_PATH)
    run = _create_result_directory(
        Path(args.out), "长期记忆有无对照_真实Embedding+真实LLM_" + _now().strftime("%Y-%m-%d_%H-%M-%S"), automatic=True,
    )
    dump(run / "cases.json", data)
    dump(run / "metadata.json", {
        "protocol": PROTOCOL, "time": _now().isoformat(), "cases_version": data["version"],
        "model": Config.LLM_MODEL, "model_digest": model_info.get("digest"),
        "ollama_version": version.json(), "embedding": Config.EMBED_MODEL_PATH,
        "embedding_device": embedder.device, "device": args.device,
        "generation_options": llm.options, "timeout": args.timeout,
        "implementation": implementation_metadata(Config),
        "cases_sha256": source_fingerprint([Path(args.cases)], Path(args.cases).parent),
        "scope": "controlled_read_side_not_end_to_end", "arm_order": "alternating",
    })
    (run / "details.jsonl").touch()
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"结果目录：{run}", flush=True)
    with (run / "details.jsonl").open("a", encoding="utf-8") as output:
        for number, case in enumerate(data["cases"]):
            runtime = None
            setup_error, state, trace, messages = None, {}, {}, {}
            with tempfile.TemporaryDirectory(prefix="memory-ablation-", dir=TEMP_ROOT) as temporary:
                root = Path(temporary)
                try:
                    runtime = build_product_runtime(
                        engine_factory=lambda: build_engine(root / "memory", root / "base", embed_model=embedder, llm=llm),
                        db_path=root / "product.db", documents_path=root / "documents", workspace_id="ablation",
                    )
                    state = seed_case(runtime, case)
                    messages, trace = paired_messages(runtime.engine, case)
                except Exception as exc:
                    setup_error = f"setup: {type(exc).__name__}: {exc}"
                try:
                    for arm in (ARMS if number % 2 == 0 else ARMS[::-1]):
                        row = {"case_id": case["id"], "arm": arm, "kind": case["kind"], "state": state,
                               "trace": trace, "messages": messages.get(arm, []), "answer": ""}
                        started = time.perf_counter()
                        if setup_error:
                            row["error"] = setup_error
                        else:
                            try:
                                row["answer"] = llm.chat(messages[arm])
                                row["generation"] = dict(llm.last_response_metadata)
                            except Exception as exc:
                                row["error"] = f"{type(exc).__name__}: {exc}"
                        row["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
                        row["answer_sha256"] = answer_hash(row["answer"])
                        row["keyword_check"] = keyword_check(case, row["answer"], row.get("error"))
                        output.write(json.dumps(row, ensure_ascii=False) + "\n")
                        output.flush()
                        print(f"[{number+1}/{len(data['cases'])}] {case['id']} {arm}: {row['answer'] or row.get('error')}", flush=True)
                finally:
                    if runtime:
                        runtime.close()
            summarize(run)
    print(f"完成：{run / 'report.md'}；规则检查仅供筛查，仍需逐条语义复核。", flush=True)
    summary = summarize(run)
    return 1 if any(m[a]["errors"] for m in summary["metrics"].values() for a in ARMS) else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="同题有无长期记忆对照（读取侧）")
    parser.add_argument("--cases", type=_local_path, default=ROOT / "cases/memory_ablation.json")
    parser.add_argument("--out", type=_local_path, default=RESULTS_ROOT)
    parser.add_argument("--device", choices=("auto", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--summarize", type=_local_path, help="仅重建已有报告，读取同目录 review.json，不调用模型")
    args = parser.parse_args(argv)
    if args.summarize:
        summary = summarize(args.summarize)
        print(json.dumps(summary["metrics"], ensure_ascii=False, indent=2))
        return 0
    return run_evaluation(args)


if __name__ == "__main__":
    raise SystemExit(main())
