"""管理用户 TXT/Markdown 资料生命周期的轻量组件，无额外依赖。

文档目录清单与向量索引相互独立。目录清单是权威来源，保留足够的分块文本，
以便文档更新或删除后重建索引。项目的 VectorStore 支持原子性整体替换；
不支持该操作的其他存储实现，可提供用于创建新存储的工厂函数。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Iterable, Optional, Union
from uuid import NAMESPACE_URL, uuid5


_MANIFEST_VERSION = 1
_MANAGED_BY = "document_lifecycle"
_SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown"}


class RebuildRequiredError(RuntimeError):
    """注入的存储无法安全删除或替换已索引的记录。"""


@dataclass(frozen=True)
class DocumentRecord:
    """持久化目录清单中的一份有效来源文档。"""

    source_id: str
    document_id: str
    title: str
    path: str
    checksum: str
    chunks: tuple[str, ...]

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "DocumentRecord":
        required = {
            "source_id",
            "document_id",
            "title",
            "path",
            "checksum",
            "chunks",
        }
        missing = required.difference(payload)
        if missing:
            raise ValueError(
                "document catalog record is missing: "
                + ", ".join(sorted(missing))
            )
        chunks = payload["chunks"]
        if not isinstance(chunks, list) or not all(
            isinstance(item, str) for item in chunks
        ):
            raise ValueError("document catalog chunks must be a string list")
        return cls(
            source_id=str(payload["source_id"]),
            document_id=str(payload["document_id"]),
            title=str(payload["title"]),
            path=str(payload["path"]),
            checksum=str(payload["checksum"]),
            chunks=tuple(chunks),
        )


@dataclass(frozen=True)
class DocumentChange:
    """一次导入操作的结果。"""

    action: str
    document: DocumentRecord
    duplicate_of: Optional[str] = None


@dataclass(frozen=True)
class SyncResult:
    """将目录清单同步到向量存储的结果。"""

    desired_count: int
    added: int
    metadata_updated: int
    replaced_records: int
    rebuilt: bool
    vector_store: Any


class DocumentLifecycle:
    """管理小型本地文档目录清单及其派生向量索引。

    通过参数注入依赖，便于在测试中用轻量替身替换实际使用的嵌入模型、
    分句器和 FAISS VectorStore。只有存储未实现 replace_all 时，
    才需要提供 vector_store_factory。
    """

    def __init__(
        self,
        manifest_path: Union[os.PathLike[str], str],
        embedder: Any,
        splitter: Any,
        vector_store: Any = None,
        vector_store_factory: Optional[Callable[[], Any]] = None,
    ):
        self.manifest_path = Path(manifest_path)
        self.embedder = embedder
        self.splitter = splitter
        self.vector_store = vector_store
        self.vector_store_factory = vector_store_factory
        self._lock = RLock()
        self._documents = self._load_manifest()

    def import_document(
        self,
        path: Union[os.PathLike[str], str],
        *,
        source_id: Optional[str] = None,
        title: Optional[str] = None,
    ) -> DocumentChange:
        """导入或更新一份 UTF-8 编码的 TXT/Markdown 来源文档。

        同一 source_id 在更新时保留原有 document_id。
        若另一来源的规范化内容完全相同，则报告重复，不再重复建立索引。
        """

        source_path = Path(path).expanduser().resolve()
        suffix = source_path.suffix.lower()
        if suffix not in _SUPPORTED_SUFFIXES:
            raise ValueError("only .txt, .md and .markdown documents are supported")
        if not source_path.is_file():
            raise FileNotFoundError(str(source_path))

        try:
            raw_text = source_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("document must be valid UTF-8") from exc

        text = _normalize_text(raw_text)
        if not text:
            raise ValueError("document is empty")
        chunks = tuple(
            chunk.strip()
            for chunk in self._split(text)
            if isinstance(chunk, str) and chunk.strip()
        )
        if not chunks:
            raise ValueError("splitter produced no non-empty chunks")

        resolved_source_id = (
            _default_source_id(source_path)
            if source_id is None
            else source_id.strip()
        )
        if not resolved_source_id:
            raise ValueError("source_id cannot be empty")
        resolved_title = source_path.stem if title is None else title.strip()
        if not resolved_title:
            raise ValueError("title cannot be empty")
        checksum = sha256(text.encode("utf-8")).hexdigest()

        with self._lock:
            existing = self._documents.get(resolved_source_id)
            duplicate = next(
                (
                    record
                    for other_source, record in self._documents.items()
                    if other_source != resolved_source_id
                    and record.checksum == checksum
                ),
                None,
            )
            if duplicate is not None:
                return DocumentChange(
                    action="duplicate",
                    document=duplicate,
                    duplicate_of=duplicate.document_id,
                )

            document_id = (
                existing.document_id
                if existing is not None
                else uuid5(
                    NAMESPACE_URL,
                    f"discrete-memory:{resolved_source_id}",
                ).hex
            )
            record = DocumentRecord(
                source_id=resolved_source_id,
                document_id=document_id,
                title=resolved_title,
                path=str(source_path),
                checksum=checksum,
                chunks=chunks,
            )
            action = "created"
            if existing is not None:
                action = "unchanged" if existing == record else "updated"
            if action != "unchanged":
                previous_documents = dict(self._documents)
                self._documents[resolved_source_id] = record
                try:
                    self._save_manifest()
                except Exception:
                    self._documents = previous_documents
                    raise
            return DocumentChange(action=action, document=record)

    def delete_document(
        self,
        *,
        source_id: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> bool:
        """从目录清单中删除一份文档。

        必须且只能提供一个标识。随后调用 :meth:`sync`，
        重建仍包含已删除分块的索引。
        """

        if bool(source_id) == bool(document_id):
            raise ValueError("provide exactly one of source_id or document_id")
        with self._lock:
            target_source = source_id
            if document_id is not None:
                target_source = next(
                    (
                        source
                        for source, record in self._documents.items()
                        if record.document_id == document_id
                    ),
                    None,
                )
            if target_source not in self._documents:
                return False
            previous_documents = dict(self._documents)
            del self._documents[str(target_source)]
            try:
                self._save_manifest()
            except Exception:
                self._documents = previous_documents
                raise
            return True

    def list_documents(self) -> list[DocumentRecord]:
        with self._lock:
            return sorted(
                self._documents.values(),
                key=lambda item: (item.title.casefold(), item.source_id),
            )

    def get_document(
        self,
        *,
        source_id: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> Optional[DocumentRecord]:
        if bool(source_id) == bool(document_id):
            raise ValueError("provide exactly one of source_id or document_id")
        with self._lock:
            if source_id is not None:
                return self._documents.get(source_id)
            return next(
                (
                    record
                    for record in self._documents.values()
                    if record.document_id == document_id
                ),
                None,
            )

    def sync(self) -> SyncResult:
        """将目录清单中的有效分块同步到注入的向量存储。

        新增分块和仅涉及元数据的变更采用增量更新。
        内容更新和删除使用存储的原子性 replace_all 操作；
        对于更简单的存储实现，保留新建存储的工厂函数作为兼容方案。
        """

        if self.vector_store is None:
            raise ValueError("vector_store is required for synchronization")

        with self._lock:
            desired = self._desired_chunks()
            existing = self._list_records(self.vector_store)
            managed = [
                item
                for item in existing
                if item.get("metadata", {}).get("managed_by") == _MANAGED_BY
            ]
            unmanaged = [
                item
                for item in existing
                if item.get("metadata", {}).get("managed_by") != _MANAGED_BY
            ]

            existing_by_id: dict[str, dict[str, Any]] = {}
            duplicate_record_ids: list[int] = []
            for item in managed:
                chunk_id = item.get("metadata", {}).get("chunk_id")
                if not chunk_id or chunk_id in existing_by_id:
                    duplicate_record_ids.append(int(item["record_id"]))
                    continue
                existing_by_id[str(chunk_id)] = item

            desired_ids = {item["metadata"]["chunk_id"] for item in desired}
            stale_ids = [
                int(item["record_id"])
                for chunk_id, item in existing_by_id.items()
                if chunk_id not in desired_ids
            ]
            requires_rebuild = bool(
                unmanaged or duplicate_record_ids or stale_ids
            )
            if requires_rebuild:
                if hasattr(self.vector_store, "replace_all"):
                    return self._replace_in_place(
                        desired,
                        replaced_records=len(existing),
                    )
                if self.vector_store_factory is None:
                    reason = (
                        "the vector store contains legacy/unmanaged or stale "
                        "knowledge records"
                    )
                    raise RebuildRequiredError(
                        reason
                        + "; provide vector_store_factory returning an empty store"
                    )
                return self._rebuild(desired, replaced_records=len(existing))

            desired_by_id = {
                item["metadata"]["chunk_id"]: item for item in desired
            }
            updates: dict[int, dict[str, Any]] = {}
            for chunk_id, item in existing_by_id.items():
                wanted = desired_by_id.get(chunk_id)
                if wanted is None:
                    continue
                current_metadata = item.get("metadata", {})
                changed = {
                    key: value
                    for key, value in wanted["metadata"].items()
                    if current_metadata.get(key) != value
                }
                if changed:
                    updates[int(item["record_id"])] = changed
            if updates:
                self.vector_store.update_metadatas(updates)

            missing = [
                item
                for chunk_id, item in desired_by_id.items()
                if chunk_id not in existing_by_id
            ]
            if missing:
                texts = [item["text"] for item in missing]
                self.vector_store.add_texts(
                    texts,
                    self._embed(texts),
                    metadatas=[item["metadata"] for item in missing],
                )

            return SyncResult(
                desired_count=len(desired),
                added=len(missing),
                metadata_updated=len(updates),
                replaced_records=0,
                rebuilt=False,
                vector_store=self.vector_store,
            )

    def rebuild(self) -> SyncResult:
        """将目录清单中的所有有效分块写入一个新的空向量存储。"""

        with self._lock:
            desired = self._desired_chunks()
            previous_count = (
                len(self._list_records(self.vector_store))
                if self.vector_store is not None
                else 0
            )
            if self.vector_store is not None and hasattr(
                self.vector_store, "replace_all"
            ):
                return self._replace_in_place(
                    desired,
                    replaced_records=previous_count,
                )
        if self.vector_store_factory is None:
            raise RebuildRequiredError(
                "provide vector_store_factory returning an empty store"
            )
        with self._lock:
            return self._rebuild(
                desired,
                replaced_records=previous_count,
            )

    def _replace_in_place(
        self,
        desired: list[dict[str, Any]],
        *,
        replaced_records: int,
    ) -> SyncResult:
        texts = [item["text"] for item in desired]
        embeddings = self._embed(texts) if texts else []
        self.vector_store.replace_all(
            texts,
            embeddings,
            metadatas=[item["metadata"] for item in desired],
        )
        return SyncResult(
            desired_count=len(desired),
            added=len(desired),
            metadata_updated=0,
            replaced_records=replaced_records,
            rebuilt=True,
            vector_store=self.vector_store,
        )

    def _rebuild(
        self,
        desired: list[dict[str, Any]],
        *,
        replaced_records: int,
    ) -> SyncResult:
        new_store = self.vector_store_factory()
        if new_store is None:
            raise ValueError("vector_store_factory returned None")
        if self._list_records(new_store):
            raise ValueError("vector_store_factory must return an empty store")
        if desired:
            texts = [item["text"] for item in desired]
            new_store.add_texts(
                texts,
                self._embed(texts),
                metadatas=[item["metadata"] for item in desired],
            )
        self.vector_store = new_store
        return SyncResult(
            desired_count=len(desired),
            added=len(desired),
            metadata_updated=0,
            replaced_records=replaced_records,
            rebuilt=True,
            vector_store=new_store,
        )

    def _desired_chunks(self) -> list[dict[str, Any]]:
        desired: list[dict[str, Any]] = []
        records = sorted(
            self._documents.values(),
            key=lambda item: (item.source_id, item.document_id),
        )
        for record in records:
            chunk_count = len(record.chunks)
            for chunk_index, text in enumerate(record.chunks):
                chunk_id = sha256(
                    (
                        f"{record.document_id}:{record.checksum}:"
                        f"{chunk_index}"
                    ).encode("utf-8")
                ).hexdigest()
                desired.append(
                    {
                        "text": text,
                        "metadata": {
                            "kind": "knowledge",
                            "status": "active",
                            "managed_by": _MANAGED_BY,
                            "slot": (
                                f"knowledge.{record.document_id}.{chunk_index}"
                            ),
                            "source_id": record.source_id,
                            "document_id": record.document_id,
                            "title": record.title,
                            "path": record.path,
                            "checksum": record.checksum,
                            "chunk_id": chunk_id,
                            "chunk_index": chunk_index,
                            "chunk_count": chunk_count,
                        },
                    }
                )
        return desired

    def _split(self, text: str) -> Iterable[str]:
        if hasattr(self.splitter, "split_text"):
            return self.splitter.split_text(text)
        if callable(self.splitter):
            return self.splitter(text)
        raise TypeError("splitter must be callable or expose split_text")

    def _embed(self, texts: list[str]) -> Any:
        if hasattr(self.embedder, "get_embeddings"):
            return self.embedder.get_embeddings(texts)
        if callable(self.embedder):
            return self.embedder(texts)
        raise TypeError("embedder must be callable or expose get_embeddings")

    @staticmethod
    def _list_records(vector_store: Any) -> list[dict[str, Any]]:
        if vector_store is None:
            return []
        if not hasattr(vector_store, "list_records"):
            raise TypeError("vector_store must expose list_records")
        return list(vector_store.list_records())

    def _load_manifest(self) -> dict[str, DocumentRecord]:
        if not self.manifest_path.exists():
            return {}
        try:
            payload = json.loads(
                self.manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid document catalog") from exc
        if not isinstance(payload, dict) or payload.get("version") != _MANIFEST_VERSION:
            raise ValueError("unsupported document catalog version")
        raw_documents = payload.get("documents", [])
        if not isinstance(raw_documents, list):
            raise ValueError("document catalog documents must be a list")

        documents: dict[str, DocumentRecord] = {}
        document_ids: set[str] = set()
        for item in raw_documents:
            if not isinstance(item, dict):
                raise ValueError("document catalog record must be an object")
            record = DocumentRecord.from_payload(item)
            if record.source_id in documents or record.document_id in document_ids:
                raise ValueError("document catalog contains duplicate identifiers")
            documents[record.source_id] = record
            document_ids.add(record.document_id)
        return documents

    def _save_manifest(self) -> None:
        payload = {
            "version": _MANIFEST_VERSION,
            "documents": [
                asdict(record)
                for record in sorted(
                    self._documents.values(),
                    key=lambda item: item.source_id,
                )
            ],
        }
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.manifest_path.with_name(
            self.manifest_path.name + ".tmp"
        )
        try:
            with temp_path.open("w", encoding="utf-8", newline="\n") as target:
                json.dump(payload, target, ensure_ascii=False, indent=2)
                target.flush()
                os.fsync(target.fileno())
            os.replace(temp_path, self.manifest_path)
        finally:
            if temp_path.exists():
                temp_path.unlink()


def _normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff").strip()


def _default_source_id(path: Path) -> str:
    return os.path.normcase(os.path.normpath(str(path)))
