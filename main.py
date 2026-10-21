"""CLI, demo and local product-server entry for DiscreteMemory."""

import argparse
import json
from pathlib import Path

from config import Config
from product_store import ProductStore


def print_product_memories(records):
    visible = [
        {
            "memory_id": item.get("memory_id"),
            "slot": item.get("slot"),
            "value": item.get("value"),
            "memory_type": item.get("memory_type"),
            "version": item.get("version"),
            "status": item.get("status"),
        }
        for item in records
    ]
    if not visible:
        print("当前没有 active 长期记忆。")
        return
    print(json.dumps(visible, ensure_ascii=False, indent=2))


def print_memory_candidates(records):
    visible = [
        {
            "candidate_id": item.get("candidate_id"),
            "slot": item.get("slot"),
            "value": item.get("value"),
            "reason_code": item.get("reason_code"),
            "policy_version": item.get("policy_version"),
            "status": item.get("status"),
        }
        for item in records
    ]
    if not visible:
        print("当前没有 pending 待确认记忆。")
        return
    print(json.dumps(visible, ensure_ascii=False, indent=2))


def _open_demo_runtime(root, resources=None):
    """Compose the real product path against isolated temporary stores."""

    from product.runtime import build_engine, build_product_runtime

    root = Path(root)
    resources = resources or {}

    def engine_factory():
        return build_engine(
            memory_db_path=str(root / "memory"),
            base_db_path=str(root / "base"),
            embed_model=resources.get("embedder"),
            llm=resources.get("llm"),
        )

    return build_product_runtime(
        engine_factory=engine_factory,
        db_path=root / "product.db",
        documents_path=root / "documents",
        workspace_id="demo",
    )


def run_demo():
    """Exercise natural conversations and the real memory lifecycle."""
    from eval.conversation_demo import run_conversation_demo

    resources = {}

    def factory(root):
        runtime = _open_demo_runtime(root, resources)
        resources.update(embedder=runtime.engine.mm.embed_model, llm=runtime.engine.llm)
        return runtime

    report = run_conversation_demo(factory)
    if report["passed_steps"] != report["steps"]:
        raise SystemExit(1)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="DiscreteMemory 本地演示")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--demo",
        action="store_true",
        help="在临时记忆库中运行一键 Demo",
    )
    group.add_argument(
        "--list-memories",
        action="store_true",
        help="查看产品数据库中的 active 离散记忆，不加载模型",
    )
    group.add_argument(
        "--serve",
        action="store_true",
        help="启动本地 FastAPI 产品与网页界面",
    )
    group.add_argument(
        "--import-legacy-memories",
        action="store_true",
        help="一次性将旧向量记忆导入全新 SQLite workspace；请先备份旧索引",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.import_legacy_memories:
        from product.runtime import build_product_runtime

        runtime = build_product_runtime(import_legacy_memories=True)
        try:
            print(json.dumps(runtime.startup, ensure_ascii=False, indent=2))
        finally:
            runtime.close()
        return
    if args.list_memories:
        with ProductStore(
            Config.PRODUCT_DB_PATH,
            default_workspace_id=Config.DEFAULT_WORKSPACE_ID,
        ) as store:
            print_product_memories(
                store.list_memories(
                    workspace_id=Config.DEFAULT_WORKSPACE_ID,
                    limit=10_000,
                )
            )
        return
    if args.demo:
        run_demo()
        return
    if args.serve:
        import uvicorn

        from api.app import create_app

        uvicorn.run(
            create_app(),
            host=Config.API_HOST,
            port=Config.API_PORT,
        )
        return

    print("\n" + "=" * 50)
    print(" DiscreteMemory：写入受治理的离散长期记忆系统")
    print("=" * 50)

    try:
        from product.runtime import build_product_runtime

        runtime = build_product_runtime()
        application = runtime.application
        session = application.create_session("CLI 会话")
        print(
            f"active 长期记忆: {len(application.list_memories(limit=10_000))}"
        )
        print(
            "输入 /memories 查看 active 记忆，/candidates 查看待确认记忆，"
            "/confirm <id> 或 /reject <id> 处理候选；输入 exit 或 quit 退出。"
        )

        try:
            while True:
                query = input("\n[用户]: ").strip()
                if not query:
                    continue
                if query.lower() in {"exit", "quit", "退出"}:
                    break
                if query.lower() == "/memories":
                    print_product_memories(application.list_memories(limit=10_000))
                    continue
                if query.lower() == "/candidates":
                    print_memory_candidates(
                        application.list_memory_candidates(
                            status="pending", limit=10_000
                        )
                    )
                    continue
                command, _, candidate_id = query.partition(" ")
                if command.lower() in {"/confirm", "/reject"}:
                    candidate_id = candidate_id.strip()
                    if not candidate_id:
                        print(f"用法：{command.lower()} <candidate_id>")
                        continue
                    try:
                        if command.lower() == "/confirm":
                            outcome = application.confirm_memory_candidate(
                                candidate_id
                            )
                            print(
                                "候选已确认："
                                + json.dumps(outcome, ensure_ascii=False)
                            )
                        else:
                            outcome = application.reject_memory_candidate(
                                candidate_id
                            )
                            print(
                                "候选已拒绝："
                                + json.dumps(outcome, ensure_ascii=False)
                            )
                    except Exception as exc:
                        print(f"[错误]: {exc}")
                    continue

                try:
                    result = application.chat(session["session_id"], query)
                    print(f"[AI]: {result['answer']}")
                    notice = result["assistant_message"]["metadata"].get("memory_notice")
                    if notice:
                        print(f"[记忆]: {notice}")
                    print(
                        "[Tool Trace]: "
                        + json.dumps(result["trace"], ensure_ascii=False)
                    )
                except Exception as exc:
                    print(f"[错误]: {exc}")
        finally:
            runtime.close()
    except KeyboardInterrupt:
        print("\n已停止。")


if __name__ == "__main__":
    main()
