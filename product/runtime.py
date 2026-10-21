"""Assemble models, retrieval indexes and application services at startup."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from config import Config
from memory.model_extractor import ModelMemoryExtractor
from knowledge.document_lifecycle import DocumentLifecycle
from product.application import ProductApplication
from product.job_fit import JobFitService
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
    """Build the read-only RAG engine, loading models only when requested."""
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
    # Knowledge only enters through the document catalog.  Do not silently
    # seed an empty projection with material absent from the documents page.
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
    """Build the real local runtime once at API or CLI startup."""

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
        # An existing empty catalog is meaningful: it can be the durable result
        # of deleting the last document before an interrupted index rebuild.
        if documents.manifest_path.exists():
            document_sync = documents.sync()
            manager.base_vs = document_sync.vector_store
        elif manager.base_vs.total_count:
            # Preserve old snapshots, but never serve unlisted documents.
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
            job_fit_service=JobFitService(llm),
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
