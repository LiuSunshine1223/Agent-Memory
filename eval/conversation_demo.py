"""在隔离状态下运行真实对话与记忆流程，并保留验证证据。

由 main.py --demo 调用。状态检查与简单回答检查分别报告；
保留原始回答供人工复核，包括失败用例的回答。
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import re
from pathlib import Path
import tempfile
import time


def _checks(app, expected, answer=None):
    states, answers = [], []
    active = app.list_memories(limit=10_000)
    pending = app.list_memory_candidates(limit=10_000)
    for label, rows in (("active", active), ("pending", pending)):
        if label + "_count" in expected:
            states.append((label + " 数量", len(rows) == expected[label + "_count"]))
        for wanted in expected.get(label, []):
            states.append((f"{label}: {wanted}", any(
                row["slot"] == wanted["slot"] and wanted["value"] in str(row["value"])
                and ("version" not in wanted or row["version"] == wanted["version"])
                for row in rows)))
    vectors = {(row["memory_id"], row["slot"], row["value"])
               for row in app.manager.list_active_memory_records(limit=10_000)}
    states.append(("SQLite / FAISS active 一致", vectors == {
        (row["memory_id"], row["slot"], row["value"]) for row in active}))
    if answer is not None:
        answers.append(("回答非空", bool(answer.strip())))
        chinese = len(re.findall(r"[\u4e00-\u9fff]", answer))
        prose = re.sub(r"\b(?:DiscreteMemory|AI|LLM|NLP|RAG|BGE|FAISS|SQLite|Python)\b", "", answer, flags=re.I)
        latin = len(re.findall(r"[A-Za-z]", prose))
        answers.append(("中文回答（允许技术名词）", chinese > 0 and chinese * 2 >= latin))
        for term in expected.get("answer_all", []):
            answers.append((f"回答包含 {term}", term.lower() in answer.lower()))
        if expected.get("answer_any"):
            terms = expected["answer_any"]
            answers.append((f"回答至少包含一个 {terms}", any(term.lower() in answer.lower() for term in terms)))
        for term in expected.get("answer_excludes", []):
            answers.append((f"回答不包含 {term}", term.lower() not in answer.lower()))
        if "answer_min_chars" in expected:
            answers.append(("回答展开请求而非只复述", len(answer) >= expected["answer_min_chars"]))
    return states, answers


def run_conversation_demo(runtime_factory, *, output_root=None, cases_path=None):
    root = Path(__file__).resolve().parent
    cases_path = Path(cases_path or root / "cases/conversation_demo.json")
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    output_root = Path(output_root or root / "results")
    output_root.mkdir(parents=True, exist_ok=True)
    run = output_root / ("对话Demo_真实Embedding+真实LLM_" + datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S"))
    run.mkdir()  # 不覆盖之前的运行结果。
    temp_root = root / "tmp"
    temp_root.mkdir(exist_ok=True)
    records = []
    print("Demo 使用隔离 SQLite/FAISS；confirm_all 模拟用户确认，不操作日常数据。")
    for scenario in cases["scenarios"]:
        print(f"\n=== {scenario['title']} ===", flush=True)
        with tempfile.TemporaryDirectory(prefix="conversation-", dir=temp_root) as temporary:
            runtime = runtime_factory(Path(temporary))
            try:
                app = runtime.application
                sid = app.create_session(scenario["title"])["session_id"]
                for index, step in enumerate(scenario["steps"], 1):
                    record = {"scenario": scenario["id"], "step": index, "input": step}
                    started = time.perf_counter()
                    try:
                        answer = None
                        if "user" in step:
                            print("用户：" + step["user"], flush=True)
                            result = app.chat(sid, step["user"])
                            answer = result["answer"]
                            record.update(answer=answer, trace=result["trace"],
                                          memory_notice=result["assistant_message"]["metadata"].get("memory_notice"))
                            print("助手：" + answer, flush=True)
                            if record["memory_notice"]:
                                print("记忆提示：" + record["memory_notice"], flush=True)
                        elif step["action"] == "confirm_all":
                            candidates = app.list_memory_candidates(limit=10_000)
                            if not candidates:
                                raise AssertionError("没有候选可供用户确认")
                            for candidate in candidates:
                                app.confirm_memory_candidate(candidate["candidate_id"])
                            print("[Demo 模拟用户确认候选]", flush=True)
                        elif step["action"] == "new_session":
                            sid = app.create_session("新会话")["session_id"]
                        elif step["action"] == "restart":
                            runtime.close()
                            runtime = runtime_factory(Path(temporary))
                            app = runtime.application
                            sid = app.create_session("重启后的新会话")["session_id"]
                        elif step["action"] == "delete_slot":
                            rows = app.list_memories(slot=step["slot"])
                            if not rows:
                                raise AssertionError("没有对应记录可供用户删除")
                            for row in rows:
                                app.delete_memory(row["memory_id"])
                        else:
                            raise ValueError("Unknown Demo action")
                        state_checks, answer_checks = _checks(app, step.get("expect", {}), answer)
                        if answer is not None:
                            state_checks.append(("聊天及抽取完成", result["trace"]["status"] == "completed"))
                            state_checks.append(("使用模型候选抽取器（纯问题允许跳过调用）",
                                result["trace"].get("intent") == "memory_forget"
                                or result["trace"].get("extraction", {}).get("method") == "llm"))
                        record.update(state_checks=state_checks, answer_checks=answer_checks,
                                      active=app.list_memories(), pending=app.list_memory_candidates())
                        record["passed"] = all(ok for _, ok in state_checks + answer_checks)
                    except Exception as exc:
                        record.update(passed=False, error=f"{type(exc).__name__}: {exc}")
                    record["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
                    records.append(record)
                    print("检查：" + ("通过" if record["passed"] else "失败，请看报告明细"), flush=True)
            finally:
                runtime.close()
    from config import Config
    # 复用项目已有的指纹格式，无须重新运行评测套件。
    from eval.run_eval import implementation_metadata, source_fingerprint
    report = {
        "protocol": cases["version"], "time": datetime.now(timezone.utc).isoformat(),
        "llm": Config.LLM_MODEL, "embedding": Config.EMBED_MODEL_PATH,
        "implementation": implementation_metadata(Config),
        "cases_sha256": source_fingerprint([cases_path], cases_path.parent),
        "steps": len(records), "chat_turns": sum("answer" in row for row in records),
        "state_passed_steps": sum(not row.get("error") and all(ok for _, ok in row.get("state_checks", [])) for row in records),
        "answer_passed_turns": sum("answer" in row and all(ok for _, ok in row.get("answer_checks", [])) for row in records),
        "extraction_model_calls": sum(row.get("trace", {}).get("extraction", {}).get("model_called", False) for row in records),
        "passed_steps": sum(row["passed"] for row in records), "records": records,
    }
    (run / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 对话与记忆 Demo 核对", "", f"时间：{report['time']}；协议：{cases['version']}。",
             f"模型：{Config.LLM_MODEL}；共 {report['chat_turns']} 轮聊天、{len(records)} 步，{report['passed_steps']} 步通过。", "",
             f"状态检查：{report['state_passed_steps']}/{len(records)}；有限回答检查：{report['answer_passed_turns']}/{report['chat_turns']}；模型抽取调用：{report['extraction_model_calls']} 次。", "",
             "检查真实应用状态与有限的回答关键词；通过不等于完整语义正确。请结合下方原始回答人工复核。",
             "包含公开开发用例，不是盲测。记忆提取失败、模型失败或不符合预期的结果均保留。", ""]
    for row in records:
        lines.extend([f"## {row['scenario']} / {row['step']} — {'通过' if row['passed'] else '失败'}", "",
                      str(row["input"].get("user", row["input"].get("action"))), ""])
        if "answer" in row:
            lines.extend([row["answer"], "", "记忆提示：" + str(row.get("memory_notice") or "无"), ""])
        failed = [label for label, ok in row.get("state_checks", []) + row.get("answer_checks", []) if not ok]
        if failed or row.get("error"):
            lines.extend(["未通过：" + str(failed or row["error"]), ""])
    (run / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n结果：{report['passed_steps']}/{len(records)}；报告：{run / 'report.md'}", flush=True)
    return report
