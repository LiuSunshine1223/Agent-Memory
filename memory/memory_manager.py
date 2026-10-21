"""Discrete long-term memory lifecycle and dual-store retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import re
from time import perf_counter
from uuid import uuid4

import numpy as np

from memory.constraints import MAX_MEMORY_VALUE_CHARS

from memory.extractor import DiscreteMemoryExtractor


@dataclass(frozen=True)
class MemoryOperation:
    action: str
    slot: str
    value: str
    version: int
    memory_id: str | None = None


class MemoryManager:
    """Manage short-term dialogue and structured long-term user facts."""

    _RETRIEVAL_TEXT_TEMPLATES = {
        "profile.name": "用户的名字是{value}",
        "profile.school": "用户就读于{value}",
        "career.goal": "用户的目标是{value}",
        "career.plan": "用户的计划是{value}",
        "preference.like": "用户喜欢{value}",
        "preference.dislike": "用户不喜欢{value}",
        "research.topic": "用户的研究方向是{value}",
        "project.focus": "用户的项目核心是{value}",
    }
    _VALID_CARDINALITIES = {"scalar", "collection", "event"}
    _VALID_SLOT = re.compile(
        r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$"
    )

    def __init__(
        self,
        base_vs,
        memory_vs,
        embed_model,
        base_max_dist,
        memory_max_dist,
        base_recall_k,
        memory_recall_k,
        short_term_limit=5,
        extractor=None,
        workspace_id="local",
    ):
        self.base_vs = base_vs
        self.memory_vs = memory_vs
        self.embed_model = embed_model
        self.base_max_dist = base_max_dist
        self.memory_max_dist = memory_max_dist
        self.base_recall_k = base_recall_k
        self.memory_recall_k = int(memory_recall_k)
        if self.memory_recall_k <= 0:
            raise ValueError("memory_recall_k must be positive")
        self.short_term_limit = int(short_term_limit)
        self.short_term_history = []
        self.extractor = extractor or DiscreteMemoryExtractor()
        self.workspace_id = str(workspace_id or "local").strip()
        self.last_retrieval_trace = None
        self._backfill_legacy_workspace()

    def retrieval_configuration(self):
        """Return safe runtime information for health checks and demos."""

        return {
            "strategy": "dense",
            "final_k": self.memory_recall_k,
        }

    def add_to_short_term(self, query, answer):
        self.short_term_history.append({"q": query, "a": answer})
        if len(self.short_term_history) > self.short_term_limit:
            self.short_term_history = self.short_term_history[-self.short_term_limit :]

    def get_chat_context(self):
        return "".join(
            f"问：{turn['q']}\n答：{turn['a']}\n"
            for turn in self.short_term_history
        )

    def memorize_to_long_term(self, query):
        """Low-level evaluation helper, without product consent or SQLite.

        Product callers must use ProductApplication/ProductMemoryService.
        """

        return [
            self.upsert_memory(
                memory_type=fact.memory_type.value,
                slot=fact.slot,
                value=fact.value,
                cardinality=self.extractor.cardinality_for_slot(fact.slot),
                source="user",
            )
            for fact in self.extractor.extract(query)
        ]

    def upsert_memory(
        self,
        memory_type,
        slot,
        value,
        cardinality="scalar",
        source="user",
        session_id=None,
        source_message_id=None,
        confirmed=True,
        memory_id=None,
    ):
        """Create one governed memory without giving an LLM side effects.

        ``scalar`` supersedes the previous active value, ``collection`` keeps
        multiple distinct values, and ``event`` appends a chronological fact.
        """

        memory_type = str(memory_type or "").strip().lower()
        slot = str(slot or "").strip().lower()
        value = " ".join(str(value or "").split()).strip()
        cardinality = str(cardinality or "scalar").strip().lower()
        if not memory_type:
            raise ValueError("memory_type is required")
        if not self._VALID_SLOT.fullmatch(slot):
            raise ValueError("slot must use dotted lowercase identifiers")
        if not value or len(value) > MAX_MEMORY_VALUE_CHARS:
            raise ValueError(
                "memory value must contain "
                f"1 to {MAX_MEMORY_VALUE_CHARS} characters"
            )
        if cardinality not in self._VALID_CARDINALITIES:
            raise ValueError("cardinality must be scalar, collection or event")

        slot_records = self._workspace_records(slot=slot)
        active_records = [
            item
            for item in slot_records
            if item["metadata"].get("status", "active") == "active"
        ]
        if cardinality != "event":
            same_value = next(
                (
                    item
                    for item in active_records
                    if self._normalize_value(item["metadata"].get("value", ""))
                    == self._normalize_value(value)
                ),
                None,
            )
            if same_value:
                return MemoryOperation(
                    "skipped",
                    slot,
                    value,
                    int(same_value["metadata"].get("version", 1)),
                    same_value["metadata"].get("memory_id"),
                )

        version = max(
            (int(item["metadata"].get("version", 0)) for item in slot_records),
            default=0,
        ) + 1
        now = self._now()
        memory_id = str(memory_id or uuid4())
        text = self._render_memory_value(slot, value)
        vector = np.asarray(
            self.embed_model.get_embeddings([text])[0], dtype="float32"
        )
        metadata = {
            "memory_id": memory_id,
            "kind": "user_memory",
            "memory_type": memory_type,
            "slot": slot,
            "value": value,
            "cardinality": cardinality,
            "status": "active" if confirmed else "candidate",
            "version": version,
            "workspace_id": self.workspace_id,
            "session_id": session_id,
            "source_message_id": source_message_id,
            "created_at": now,
            "updated_at": now,
            "source": str(source or "user")[:50],
            "confirmed": bool(confirmed),
        }
        changed_records = []
        if cardinality == "scalar":
            for old_record in active_records:
                old_metadata = old_record["metadata"]
                # Legacy scalar preferences can coexist with new collection items.
                if old_metadata.get("cardinality", "scalar") != "scalar":
                    continue
                changed_records.append(
                    (
                        old_record["record_id"],
                        {
                            "status": old_metadata.get("status", "active"),
                            "updated_at": old_metadata.get("updated_at"),
                            "superseded_by": old_metadata.get("superseded_by"),
                        },
                    )
                )
                self.memory_vs.update_metadata(
                    old_record["record_id"],
                    {
                        "status": "superseded",
                        "updated_at": now,
                        "superseded_by": memory_id,
                    },
                    persist=False,
                )

        try:
            self.memory_vs.add_texts(
                [text],
                np.asarray([vector], dtype="float32"),
                metadatas=[metadata],
            )
        except Exception:
            for record_id, previous in changed_records:
                self.memory_vs.update_metadata(
                    record_id,
                    previous,
                    persist=False,
                )
            raise
        action = "updated" if cardinality == "scalar" and active_records else "created"
        return MemoryOperation(action, slot, value, version, memory_id)

    def edit_memory(self, memory_id, value):
        """Version an existing active memory after an explicit user edit."""

        record = self._find_memory(memory_id, active_only=True)
        if not record:
            return None
        metadata = record["metadata"]
        cardinality = metadata.get("cardinality", "scalar")
        if cardinality == "collection":
            normalized = self._normalize_value(value)
            duplicate = next(
                (
                    item
                    for item in self._workspace_records(
                        slot=metadata.get("slot"), status="active"
                    )
                    if item["metadata"].get("memory_id") != str(memory_id)
                    and self._normalize_value(
                        item["metadata"].get("value", "")
                    )
                    == normalized
                ),
                None,
            )
            if duplicate is not None:
                return MemoryOperation(
                    "skipped",
                    metadata.get("slot"),
                    value,
                    int(duplicate["metadata"].get("version", 1)),
                    duplicate["metadata"].get("memory_id"),
                )
        previous_metadata = None
        if cardinality != "scalar":
            now = self._now()
            previous_metadata = {
                "status": metadata.get("status", "active"),
                "updated_at": metadata.get("updated_at"),
            }
            self.memory_vs.update_metadata(
                record["record_id"],
                {"status": "superseded", "updated_at": now},
                persist=False,
            )
        try:
            return self.upsert_memory(
                memory_type=metadata.get("memory_type", "fact"),
                slot=metadata.get("slot"),
                value=value,
                cardinality=cardinality,
                source="user_edit",
                session_id=metadata.get("session_id"),
                confirmed=True,
                memory_id=memory_id,
            )
        except Exception:
            if previous_metadata is not None:
                self.memory_vs.update_metadata(
                    record["record_id"],
                    previous_metadata,
                    persist=False,
                )
            raise

    def delete_memory(self, memory_id):
        """Logically delete one active memory while retaining its audit trail."""

        record = self._find_memory(memory_id, active_only=True)
        if not record:
            return False
        self.memory_vs.update_metadata(
            record["record_id"],
            {
                "status": "deleted",
                "updated_at": self._now(),
                "deleted_reason": "explicit_user_request",
            },
        )
        return True

    def purge_memory(self, memory_id):
        """Physically remove one memory from metadata and its FAISS index."""

        if not hasattr(self.memory_vs, "purge_records"):
            raise RuntimeError("the configured vector store cannot purge records")
        return self.memory_vs.purge_records(
            {
                "memory_id": str(memory_id),
                "workspace_id": self.workspace_id,
                "kind": "user_memory",
            }
        )

    def export_memories(self):
        """Return all memory versions for the current workspace."""

        records = []
        for item in self._workspace_records():
            exported = dict(item["metadata"])
            exported["text"] = item["text"]
            records.append(exported)
        records.sort(
            key=lambda item: (
                item.get("slot", ""),
                int(item.get("version", 0)),
            )
        )
        return records

    def forget(self, query):
        """Low-level vector-only helper for tests/evaluation, not product chat."""
        slots = self.extractor.infer_forget_slots(query)
        if not slots:
            return 0

        active_records = self._workspace_records(status="active")
        selected = [
            item
            for item in active_records
            if "*" in slots or item["metadata"].get("slot") in slots
        ]
        if not selected:
            return 0

        now = self._now()
        updates = {
            item["record_id"]: {
                "status": "deleted",
                "updated_at": now,
                "deleted_reason": "explicit_user_request",
            }
            for item in selected
        }
        self.memory_vs.update_metadatas(updates)
        return len(selected)

    def list_active_memory_records(self, limit=100):
        records = self._workspace_records(status="active")
        result = []
        for item in records:
            metadata = dict(item["metadata"])
            metadata["record_id"] = item["record_id"]
            metadata["text"] = item["text"]
            result.append(metadata)
        result.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
        return result[:limit]

    def get_memory_recall_context(self, query):
        slots = self.extractor.infer_recall_slots(query)
        if slots:
            memory_docs = self._lookup_active_memory_slots(slots)
            retrieval_mode = "slot_lookup"
        elif self.extractor.is_list_memory_request(query):
            direct_started_at = perf_counter()
            active_records = self.list_active_memory_records()
            memory_docs = [item["text"] for item in active_records]
            retrieval_mode = "list_active"
            self._set_direct_retrieval_trace(
                retrieval_mode,
                active_records,
                latency_ms=self._elapsed_ms(direct_started_at),
            )
        else:
            memory_docs = self._search_memory(
                query,
                max_distance=self.memory_max_dist,
            )
            retrieval_mode = "semantic_search"

        return {
            "base_context": "无。",
            "memory_context": "\n".join(memory_docs) if memory_docs else "无。",
            "memory_records": self._selected_memory_records(),
            "memory_retrieval_mode": retrieval_mode,
        }

    def get_long_term_context(self, query):
        query_embedding = self.embed_model.get_embeddings([query])

        base_docs = []
        if self.base_vs.total_count > 0:
            base_docs = self.base_vs.search(
                query_embedding,
                self.base_recall_k,
                self.base_max_dist,
            )

        memory_docs = self._search_memory(
            query,
            max_distance=self.memory_max_dist,
            query_embedding=query_embedding,
        )
        return {
            "base_context": "\n".join(base_docs) if base_docs else "无。",
            "memory_context": "\n".join(memory_docs) if memory_docs else "无。",
            "memory_records": self._selected_memory_records(),
        }

    def _selected_memory_records(self):
        """Expose slot identity so answer-time conflicts need no text matching."""
        records = {item["record_id"]: item for item in self.list_active_memory_records(limit=10_000)}
        return [records[item["record_id"]] for item in (self.last_retrieval_trace or {}).get("candidates", [])
                if item.get("selected") and item.get("record_id") in records]

    def _search_memory(
        self,
        query,
        max_distance,
        query_embedding=None,
    ):
        records = self.retrieve_memory_records(
            query,
            max_distance=max_distance,
            query_embedding=query_embedding,
        )
        return [item["text"] for item in records]

    def retrieve_memory_records(
        self,
        query,
        max_distance=None,
        query_embedding=None,
    ):
        """Retrieve active workspace memories with dense FAISS search."""

        started_at = perf_counter()
        filters = {
            "status": "active",
            "kind": "user_memory",
            "workspace_id": self.workspace_id,
        }
        timings = {
            "embedding": 0.0,
            "dense": 0.0,
        }

        if self.memory_vs.total_count == 0:
            self._set_semantic_retrieval_trace(
                records=[],
                max_distance=max_distance,
                timings=timings,
                started_at=started_at,
            )
            return []

        if query_embedding is None:
            embedding_started_at = perf_counter()
            query_embedding = self.embed_model.get_embeddings([query])
            timings["embedding"] = self._elapsed_ms(embedding_started_at)

        dense_started_at = perf_counter()
        dense_records = self.memory_vs.search_records(
            query_embedding,
            self.memory_recall_k,
            max_distance=max_distance,
            filters=filters,
        )
        timings["dense"] = self._elapsed_ms(dense_started_at)
        returned = []
        for rank, record in enumerate(dense_records, start=1):
            item = dict(record)
            item["metadata"] = dict(record.get("metadata") or {})
            item["vector_rank"] = rank
            item["final_rank"] = rank
            returned.append(item)
        self._set_semantic_retrieval_trace(
            records=returned,
            max_distance=max_distance,
            timings=timings,
            started_at=started_at,
        )
        return returned

    def _lookup_active_memory_slots(self, slots):
        direct_started_at = perf_counter()
        records = self._workspace_records(status="active")
        docs = []
        selected_records = []
        for slot in slots:
            candidates = [
                item
                for item in records
                if item["metadata"].get("slot") == slot
            ]
            if not candidates:
                continue
            ordered = sorted(
                candidates,
                key=lambda item: int(item["metadata"].get("version", 0)),
            )
            if all(
                item["metadata"].get("cardinality", "scalar") == "scalar"
                for item in ordered
            ):
                ordered = ordered[-1:]
            docs.extend(item["text"] for item in ordered)
            selected_records.extend(ordered)
        self._set_direct_retrieval_trace(
            "slot_lookup",
            selected_records,
            latency_ms=self._elapsed_ms(direct_started_at),
        )
        return docs

    def _set_direct_retrieval_trace(self, mode, records, latency_ms=0.0):
        candidates = []
        for rank, record in enumerate(records, start=1):
            item = self._safe_retrieval_record(record, selected=True)
            item["final_rank"] = rank
            candidates.append(item)
        self.last_retrieval_trace = {
            "mode": mode,
            "strategy": "structured_metadata",
            "candidate_k": len(records),
            "final_k": len(records),
            "candidate_count": len(records),
            "returned_count": len(records),
            "latency_ms": {"total": round(float(latency_ms), 3)},
            "candidates": candidates,
        }

    def _set_semantic_retrieval_trace(
        self,
        *,
        records,
        max_distance,
        timings,
        started_at,
    ):
        safe_candidates = [
            self._safe_retrieval_record(record, selected=True)
            for record in records
        ]
        latency = {
            name: round(float(value), 3)
            for name, value in timings.items()
        }
        latency["total"] = self._elapsed_ms(started_at)
        self.last_retrieval_trace = {
            "mode": "semantic_search",
            "strategy": "dense",
            "candidate_k": self.memory_recall_k,
            "final_k": self.memory_recall_k,
            "candidate_count": len(records),
            "returned_count": len(records),
            "max_distance": (
                float(max_distance) if max_distance is not None else None
            ),
            "latency_ms": latency,
            "candidates": safe_candidates,
        }

    @staticmethod
    def _safe_retrieval_record(record, selected):
        metadata = record.get("metadata") or record
        safe = {
            "record_id": record.get("record_id"),
            "memory_id": metadata.get("memory_id"),
            "slot": metadata.get("slot"),
            "selected": bool(selected),
        }
        integer_fields = ("vector_rank", "final_rank")
        score_fields = ("distance",)
        for name in integer_fields:
            if record.get(name) is not None:
                safe[name] = int(record[name])
        for name in score_fields:
            if record.get(name) is not None:
                safe[name] = round(float(record[name]), 8)
        return safe

    @staticmethod
    def _elapsed_ms(started_at):
        return round((perf_counter() - started_at) * 1000.0, 3)

    @classmethod
    def _render_memory_text(cls, fact):
        return cls._render_memory_value(fact.slot, fact.value)

    @classmethod
    def _render_memory_value(cls, slot, value):
        template = cls._RETRIEVAL_TEXT_TEMPLATES.get(
            slot, "用户信息：{value}"
        )
        return template.format(value=value)

    def _workspace_records(self, slot=None, status=None):
        records = self.memory_vs.list_records({"kind": "user_memory"})
        selected = []
        for item in records:
            metadata = item["metadata"]
            if metadata.get("workspace_id", "local") != self.workspace_id:
                continue
            if slot is not None and metadata.get("slot") != slot:
                continue
            if status is not None and metadata.get("status", "active") != status:
                continue
            selected.append(item)
        return selected

    def _find_memory(self, memory_id, active_only=False):
        for item in self._workspace_records():
            metadata = item["metadata"]
            if metadata.get("memory_id") != str(memory_id):
                continue
            if active_only and metadata.get("status", "active") != "active":
                continue
            return item
        return None

    def _backfill_legacy_workspace(self):
        if self.workspace_id != "local":
            return
        updates = {}
        for item in self.memory_vs.list_records({"kind": "user_memory"}):
            if not item["metadata"].get("workspace_id"):
                updates[item["record_id"]] = {
                    "workspace_id": "local",
                    "cardinality": item["metadata"].get(
                        "cardinality", "scalar"
                    ),
                }
        if updates:
            self.memory_vs.update_metadatas(updates)

    @staticmethod
    def _normalize_value(value):
        return "".join(str(value).lower().split()).strip("，。！？,.!?")

    @staticmethod
    def _now():
        return datetime.now(timezone.utc).isoformat()
