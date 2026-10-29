"""轻量 FastAPI 适配层；业务行为由 product 中的服务实现。"""

from __future__ import annotations

from contextlib import asynccontextmanager
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any, Literal, Optional, Union
from uuid import NAMESPACE_URL, uuid5

from fastapi import FastAPI, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from config import Config
from knowledge.document_lifecycle import RebuildRequiredError
from memory.constraints import MAX_MEMORY_VALUE_CHARS
from models.llm import LLMUnavailableError
from product.runtime import build_product_runtime
from product_store import ConflictError, NotFoundError


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SessionCreate(_StrictModel):
    title: Optional[str] = Field(default=None, max_length=200)


class ChatRequest(_StrictModel):
    session_id: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=10_000)


def _validate_memory_value(value):
    if value is None:
        raise ValueError("value must be non-null")
    try:
        encoded = (
            value
            if isinstance(value, str)
            else json.dumps(value, ensure_ascii=False, sort_keys=True)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("value must be JSON-serializable") from exc
    if not encoded.strip() or len(encoded) > MAX_MEMORY_VALUE_CHARS:
        raise ValueError(
            "value must contain 1 to "
            f"{MAX_MEMORY_VALUE_CHARS} characters"
        )
    return value


class MemoryUpdate(_StrictModel):
    value: Any

    @field_validator("value")
    @classmethod
    def validate_value(cls, value):
        return _validate_memory_value(value)


class DocumentCreate(_StrictModel):
    source_id: str = Field(min_length=1, max_length=100)
    title: str = Field(min_length=1, max_length=200)
    filename: str = Field(default="document.md", min_length=1, max_length=200)
    content: str = Field(min_length=1, max_length=2_000_000)

    @field_validator("source_id", "title", "filename")
    @classmethod
    def strip_required_text(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("field must not be blank")
        return value


def create_app(
    application=None,
    *,
    runtime_factory=build_product_runtime,
    document_root: Optional[Union[str, Path]] = None,
    web_path: Optional[Union[str, Path]] = None,
) -> FastAPI:
    """使用注入的替身组件或真实本地运行环境创建应用。"""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime = None
        if application is None:
            runtime = runtime_factory()
            app.state.product = runtime.application
            app.state.startup = dict(runtime.startup)
        else:
            app.state.product = application
            app.state.startup = {"mode": "injected"}
        try:
            yield
        finally:
            if runtime is not None:
                runtime.close()

    app = FastAPI(
        title="DiscreteMemory",
        version="0.1.0",
        description="支持长期记忆与资料问答的对话系统",
        lifespan=lifespan,
    )
    managed_root = Path(document_root or Config.DOCUMENTS_PATH) / "sources"
    index_path = Path(web_path or Path(__file__).resolve().parents[1] / "web" / "index.html")

    @app.exception_handler(NotFoundError)
    def not_found_handler(_request: Request, exc: NotFoundError):
        return _error(404, "not_found", str(exc))

    @app.exception_handler(ConflictError)
    def conflict_handler(_request: Request, exc: ConflictError):
        return _error(409, "conflict", str(exc))

    @app.exception_handler(RebuildRequiredError)
    def rebuild_handler(_request: Request, exc: RebuildRequiredError):
        return _error(409, "index_rebuild_required", str(exc))

    @app.exception_handler(LLMUnavailableError)
    def llm_handler(_request: Request, exc: LLMUnavailableError):
        return _error(503, "llm_unavailable", str(exc))

    @app.exception_handler(ValueError)
    def value_handler(_request: Request, exc: ValueError):
        return _error(422, "invalid_request", str(exc))

    @app.get("/", include_in_schema=False)
    def index():
        if index_path.is_file():
            return FileResponse(index_path)
        return {"name": "DiscreteMemory", "docs": "/docs"}

    @app.get("/health")
    def health(request: Request):
        return {
            "status": "ok",
            "product": "DiscreteMemory",
            "startup": request.app.state.startup,
        }

    @app.post("/v1/sessions", status_code=201)
    def create_session(request: Request, payload: SessionCreate):
        return _product(request).create_session(payload.title)

    @app.get("/v1/sessions")
    def list_sessions(request: Request, limit: int = Query(100, ge=1, le=1000)):
        return {"items": _product(request).list_sessions(limit=limit)}

    @app.delete("/v1/sessions/{session_id}")
    def delete_session(request: Request, session_id: str):
        """删除会话及聊天消息；长期记忆和候选仍在记忆页面管理。"""
        return _product(request).delete_session(session_id)

    @app.get("/v1/sessions/{session_id}/messages")
    def list_messages(
        request: Request,
        session_id: str,
        limit: int = Query(1000, ge=1, le=10_000),
    ):
        return {"items": _product(request).list_messages(session_id, limit=limit)}

    @app.post("/v1/chat")
    def chat(request: Request, payload: ChatRequest):
        return _product(request).chat(payload.session_id, payload.message)

    @app.get("/v1/memories")
    def list_memories(
        request: Request,
        status: Literal["active", "deleted"] = "active",
        memory_type: Optional[Literal["scalar", "collection", "event"]] = None,
        slot: Optional[str] = None,
        limit: int = Query(100, ge=1, le=10_000),
    ):
        return {
            "items": _product(request).list_memories(
                status=status,
                memory_type=memory_type,
                slot=slot,
                limit=limit,
            )
        }

    @app.get("/v1/memories/{memory_id}/history")
    def memory_history(request: Request, memory_id: str):
        return {"items": _product(request).memory_history(memory_id)}

    @app.get("/v1/memory-candidates")
    def list_memory_candidates(
        request: Request,
        status: Literal["pending", "accepted", "rejected"] = "pending",
        limit: int = Query(100, ge=1, le=10_000),
    ):
        return {
            "items": _product(request).list_memory_candidates(
                status=status,
                limit=limit,
            )
        }

    @app.post("/v1/memory-candidates/{candidate_id}/confirm")
    def confirm_memory_candidate(request: Request, candidate_id: str):
        return _product(request).confirm_memory_candidate(candidate_id)

    @app.post("/v1/memory-candidates/{candidate_id}/reject")
    def reject_memory_candidate(request: Request, candidate_id: str):
        return _product(request).reject_memory_candidate(candidate_id)

    @app.patch("/v1/memories/{memory_id}")
    def update_memory(request: Request, memory_id: str, payload: MemoryUpdate):
        return _product(request).update_memory(memory_id, payload.value)

    @app.delete("/v1/memories/{memory_id}")
    def delete_memory(request: Request, memory_id: str):
        """逻辑删除长期记忆；原始聊天消息保留，仍可能进入近期对话上下文。"""
        return _product(request).delete_memory(memory_id)

    @app.delete("/v1/memories/{memory_id}/purge")
    def purge_memory(request: Request, memory_id: str):
        """清除已删除记忆的版本、关联候选和向量；不删除原始聊天消息。"""
        return {"purged": _product(request).purge_memory(memory_id)}

    @app.get("/v1/documents")
    def list_documents(request: Request):
        return {"items": _product(request).list_documents()}

    @app.post("/v1/documents", status_code=201)
    def import_document(request: Request, payload: DocumentCreate):
        product = _product(request)
        previous = next(
            (
                item
                for item in product.list_documents()
                if item["source_id"] == payload.source_id
            ),
            None,
        )
        target = _managed_document_path(
            managed_root,
            payload.source_id,
            payload.filename,
            payload.content,
        )
        _atomic_write_text(target, payload.content)
        try:
            result = product.import_document(
                target,
                source_id=payload.source_id,
                title=payload.title,
            )
            indexed_path = Path(result["document"]["path"]).resolve()
            if result.get("action") == "duplicate" and target.resolve() != indexed_path:
                target.unlink(missing_ok=True)
            elif previous and Path(previous["path"]).resolve() != indexed_path:
                _delete_if_managed(Path(previous["path"]), managed_root)
            return result
        except Exception:
            # 文档目录清单仍是权威来源；首次导入失败时，
            # 不应遗留没有被目录清单引用的上传文件。
            known_paths = {
                Path(item["path"]).resolve()
                for item in product.list_documents()
            }
            if target.resolve() not in known_paths:
                target.unlink(missing_ok=True)
            elif previous and Path(previous["path"]).resolve() not in known_paths:
                _delete_if_managed(Path(previous["path"]), managed_root)
            raise

    @app.delete("/v1/documents/{document_id}")
    def delete_document(request: Request, document_id: str):
        product = _product(request)
        record = next(
            (
                item
                for item in product.list_documents()
                if item["document_id"] == document_id
            ),
            None,
        )
        result = product.delete_document(document_id=document_id)
        if result["deleted"] and record:
            _delete_if_managed(Path(record["path"]), managed_root)
        return result


    return app


def _product(request: Request):
    return request.app.state.product


def _error(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}},
    )


def _managed_document_path(
    root: Path,
    source_id: str,
    filename: str,
    content: str,
) -> Path:
    suffix = Path(filename).suffix.lower()
    if suffix not in {".txt", ".md", ".markdown"}:
        raise ValueError("filename must end in .txt, .md or .markdown")
    stable_name = uuid5(NAMESPACE_URL, f"discrete-memory:{source_id}").hex
    revision = sha256(content.encode("utf-8")).hexdigest()[:12]
    return root / f"{stable_name}-{revision}{suffix}"


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    try:
        with temp.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _delete_if_managed(path: Path, root: Path) -> None:
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    if resolved_path.parent == resolved_root:
        resolved_path.unlink(missing_ok=True)


__all__ = ["create_app"]
