"""在启动时组装模型、检索索引和应用服务。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from config import Config
from memory.model_extractor import ModelMemoryExtractor
from knowledge.document_lifecycle import DocumentLifecycle
from product.application import ProductApplication
from product.memory_service import ProductMemoryService
from product_store import ProductStore
from utils.text_splitter import SentenceBoundaryChunker


def build_engine(
    memory_db_path=None,
    base_db_path=None,
    *,
    embed_model=None,
    llm=None,
):
    """构建只读 RAG 引擎，仅在需要时加载模型。"""
    from memory.memory_manager import MemoryManager
    from models.embedding import EmbeddingModel
    from models.llm import LLMModel
    from pipeline.rag_pipeline import RAGPipeline
    from retrieval.vector_store import VectorStore

    embed_model = embed_model or EmbeddingModel(Config.EMBED_MODEL_PATH)
    llm = llm or LLMModel(
        Config.OLLAMA_URL,
        Config.LLM_MODEL,
        timeout=Config.LLM_TIMEOUT_SECONDS,
    )
    base_vs = VectorStore(base_db_path or Config.BASE_DB_PATH, Config.VECTOR_DIM)
    # 知识资料只能通过文档目录清单进入索引，不能向空索引
    # 静默加入文档页面未列出的资料。
    memory_vs = VectorStore(memory_db_path or Config.MEMORY_DB_PATH, Config.VECTOR_DIM)

    memory_manager = MemoryManager(
        base_vs=base_vs,
        memory_vs=memory_vs,
        embed_model=embed_model,
        base_max_dist=Config.BASE_MAX_DISTANCE,
        memory_max_dist=Config.MEMORY_MAX_DISTANCE,
        base_recall_k=Config.BASE_RECALL_K,
        memory_recall_k=Config.MEMORY_RECALL_K,
        short_term_limit=Config.SHORT_TERM_LIMIT,
    )
    return RAGPipeline(llm, memory_manager), memory_vs


@dataclass
class ProductRuntime:
    application: ProductApplication
    store: ProductStore
    engine: Any
    startup: dict

    def close(self) -> None:
        self.store.close()


def build_product_runtime(
    *,
    engine_factory: Callable[[], tuple[Any, Any]] | None = None,
    db_path: str | Path | None = None,
    documents_path: str | Path | None = None,
    workspace_id: str | None = None,
    import_legacy_memories: bool = False,
) -> ProductRuntime:
    """在接口服务或命令行启动时，一次性构建真实本地运行环境。"""

    if engine_factory is None:
        engine_factory = build_engine

    workspace_id = workspace_id or Config.DEFAULT_WORKSPACE_ID
    engine, _ = engine_factory()
    manager, llm = _engine_components(engine)
    manager.workspace_id = workspace_id

    store = ProductStore(
        db_path or Config.PRODUCT_DB_PATH,
        default_workspace_id=workspace_id,
    )
    try:
        memory = ProductMemoryService(store, manager, workspace_id)
        legacy_imported, memory_reconcile = memory.initialize_index(
            import_legacy=import_legacy_memories
        )

        document_root = Path(documents_path or Config.DOCUMENTS_PATH)
        documents = DocumentLifecycle(
            manifest_path=document_root / "catalog.json",
            embedder=manager.embed_model,
            splitter=SentenceBoundaryChunker(
                Config.CHUNK_SIZE,
                Config.OVERLAP_SENTENCE,
            ),
            vector_store=manager.base_vs,
        )
        document_sync = None
        # 已存在的空目录清单具有实际含义：可能是删除最后一份文档后
        # 索引重建被中断，而删除结果已经持久化。
        if documents.manifest_path.exists():
            document_sync = documents.sync()
            manager.base_vs = document_sync.vector_store
        elif manager.base_vs.total_count:
            # 保留旧快照，但不提供目录清单之外的文档。
            raise ValueError(
                "knowledge index has no document catalog; back up the old "
                "index, use an empty BASE_DB_PATH and explicitly import "
                "TXT/Markdown sources through /v1/documents"
            )

        application = ProductApplication(
            store=store,
            memory_service=memory,
            engine=engine,
            document_lifecycle=documents,
            workspace_id=workspace_id,
            fact_extractor=ModelMemoryExtractor(llm),
        )
        return ProductRuntime(
            application=application,
            store=store,
            engine=engine,
            startup={
                "workspace_id": workspace_id,
                "legacy_memories_imported": legacy_imported,
                "memory_reconcile": memory_reconcile,
                "document_chunks": (
                    document_sync.desired_count if document_sync else 0
                ),
                "retrieval": manager.retrieval_configuration(),
            },
        )
    except Exception:
        store.close()
        raise


def _engine_components(engine):
    pipeline = getattr(engine, "pipeline", engine)
    manager = getattr(pipeline, "mm", None)
    llm = getattr(pipeline, "llm", None)
    if manager is None or llm is None:
        raise TypeError("engine must expose a pipeline with llm and memory manager")
    return manager, llm


__all__ = ["ProductRuntime", "build_engine", "build_product_runtime"]
