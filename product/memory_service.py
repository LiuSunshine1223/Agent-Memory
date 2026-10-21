"""Coordinate SQLite product state with the local FAISS memory index."""

from __future__ import annotations

from dataclasses import dataclass
import json
from uuid import uuid4

from product_store import ConflictError, NotFoundError, ProductStore


@dataclass(frozen=True)
class ProductMemoryOutcome:
    action: str
    memory: dict
    vector_action: str | None = None


class ProductMemoryService:
    """Single entry point for user-authorized memory side effects.

    SQLite is the source of truth for user-visible state and revision history.
    ``MemoryManager`` keeps a matching FAISS representation for semantic
    recall.  All product callers go through this class so the two stores stay
    aligned without exposing write or delete side effects to the LLM.
    """

    def __init__(
        self,
        store: ProductStore,
        memory_manager,
        workspace_id="local",
    ):
        self.store = store
        self.manager = memory_manager
        self.workspace_id = str(workspace_id or "local").strip()
        self.store.create_workspace(self.workspace_id)

    def remember(
        self,
        slot,
        value,
        *,
        cardinality="scalar",
        semantic_type=None,
        source="user",
        source_message_id=None,
        session_id=None,
        provenance=None,
    ):
        cardinality = str(cardinality or "scalar").strip().lower()
        semantic_type = str(
            semantic_type or str(slot).split(".", 1)[0]
        ).strip()
        provenance = {
            **dict(provenance or {}),
            "semantic_type": semantic_type,
            "index": "faiss",
        }
        existing = self.store.list_memories(
            workspace_id=self.workspace_id,
            memory_type=None if slot in self.manager.extractor.COLLECTION_SLOTS else cardinality,
            slot=slot,
            limit=10_000,
        )
        duplicate = next(
            (
                item
                for item in existing
                if self._normalize(item["value"]) == self._normalize(value)
            ),
            None,
        )
        if duplicate and cardinality != "event":
            return ProductMemoryOutcome("skipped", duplicate, "skipped")

        scalar_existing = [item for item in existing if item["memory_type"] == "scalar"]
        if cardinality == "scalar" and scalar_existing:
            current = scalar_existing[0]
            vector_attempted = False
            try:
                with self.store.transaction():
                    updated = self.store.edit_memory(
                        current["memory_id"],
                        value,
                        source=source,
                        provenance=provenance,
                        source_message_id=source_message_id,
                        workspace_id=self.workspace_id,
                    )
                    vector_attempted = True
                    vector = self.manager.upsert_memory(
                        semantic_type,
                        slot,
                        self._value_text(value),
                        cardinality=cardinality,
                        source=source,
                        session_id=session_id,
                        source_message_id=source_message_id,
                        memory_id=current["memory_id"],
                    )
            except Exception:
                if vector_attempted:
                    self._best_effort_reconcile()
                raise
            return ProductMemoryOutcome("updated", updated, vector.action)

        memory_id = str(uuid4())
        vector_attempted = False
        try:
            with self.store.transaction():
                created = self.store.create_memory(
                    slot,
                    value,
                    memory_type=cardinality,
                    source=source,
                    provenance=provenance,
                    source_message_id=source_message_id,
                    memory_id=memory_id,
                    workspace_id=self.workspace_id,
                )
                vector_attempted = True
                vector = self.manager.upsert_memory(
                    semantic_type,
                    slot,
                    self._value_text(value),
                    cardinality=cardinality,
                    source=source,
                    session_id=session_id,
                    source_message_id=source_message_id,
                    memory_id=memory_id,
                )
        except Exception:
            if vector_attempted:
                self._best_effort_reconcile()
            raise
        return ProductMemoryOutcome("created", created, vector.action)

    def remember_extracted(
        self,
        text,
        *,
        source_message_id=None,
        session_id=None,
        source="user",
        provenance=None,
        facts=None,
    ):
        write_provenance = {
            "extractor": "explicit_rule",
            **dict(provenance or {}),
        }
        return [
            self.remember(
                fact.slot,
                fact.value,
                cardinality=self.manager.extractor.cardinality_for_slot(fact.slot),
                semantic_type=fact.memory_type.value,
                source=source,
                source_message_id=source_message_id,
                session_id=session_id,
                provenance={**write_provenance,
                            "extractor": "llm" if fact.evidence else "explicit_rule",
                            "evidence": fact.evidence or text},
            )
            for fact in (self.manager.extractor.extract(text) if facts is None else facts)
        ]

    def propose_extracted(
        self,
        text,
        *,
        source_message_id=None,
        session_id=None,
        reason_code="confirmation_required",
        policy_version="consent-gated-v1",
        facts=None,
    ):
        """Stage new extracted facts without making them recallable."""

        pending = self.store.list_memory_candidates(
            workspace_id=self.workspace_id,
            status="pending",
            limit=1_000_000,
        )
        active = self.store.list_memories(
            workspace_id=self.workspace_id,
            status="active",
            limit=1_000_000,
        )
        candidates = []
        for fact in (self.manager.extractor.extract(text) if facts is None else facts):
            cardinality = self.manager.extractor.cardinality_for_slot(fact.slot)
            normalized = self._normalize(fact.value)
            slot_active = [
                item
                for item in active
                if item["slot"] == fact.slot and item["memory_type"] == "scalar" and cardinality == "scalar"
            ]
            if any(
                item["slot"] == fact.slot
                and self._normalize(item["value"]) == normalized
                for item in active
            ):
                continue
            if any(
                item["slot"] == fact.slot
                and self._normalize(item["value"]) == normalized
                for item in pending
            ):
                continue
            expected = slot_active[0] if slot_active else None
            try:
                candidate = self.store.create_memory_candidate(
                    fact.slot,
                    fact.value,
                    memory_type=cardinality,
                    semantic_type=fact.memory_type.value,
                    reason_code=reason_code,
                    policy_version=policy_version,
                    provenance={
                        "extractor": "llm" if fact.evidence else "explicit_rule",
                        "evidence": fact.evidence or text,
                        "admission": "candidate",
                        "expected_memory_id": (
                            expected["memory_id"] if expected else None
                        ),
                        "expected_memory_version": (
                            expected["version"] if expected else None
                        ),
                    },
                    session_id=session_id,
                    source_message_id=source_message_id,
                    workspace_id=self.workspace_id,
                )
            except ConflictError:
                # The partial unique index closes the race between the list
                # above and insert when two local callers propose the same fact.
                concurrent_pending = self.store.list_memory_candidates(
                    workspace_id=self.workspace_id,
                    status="pending",
                    limit=1_000_000,
                )
                if any(
                    item["slot"] == fact.slot
                    and self._normalize(item["value"]) == normalized
                    for item in concurrent_pending
                ):
                    pending = concurrent_pending
                    continue
                raise
            candidates.append(candidate)
            pending.append(candidate)
        return candidates

    def list_candidates(self, *, status="pending", limit=100):
        return self.store.list_memory_candidates(
            workspace_id=self.workspace_id,
            status=status,
            limit=limit,
        )

    def confirm_candidate(self, candidate_id):
        """Promote one pending candidate through the normal SQLite/FAISS path."""

        write_attempted = False
        try:
            with self.store.transaction():
                candidate = self.store.get_memory_candidate(
                    candidate_id, workspace_id=self.workspace_id
                )
                if candidate["status"] != "pending":
                    raise ConflictError(
                        "memory candidate is already "
                        f"{candidate['status']}: {candidate_id}"
                    )
                self._validate_candidate_snapshot(candidate)
                write_attempted = True
                outcome = self.remember(
                    candidate["slot"],
                    candidate["value"],
                    cardinality=("collection" if candidate["slot"] in self.manager.extractor.COLLECTION_SLOTS
                                 else candidate["memory_type"]),
                    semantic_type=candidate["semantic_type"],
                    source="user_confirmed",
                    source_message_id=candidate["source_message_id"],
                    session_id=candidate["session_id"],
                    provenance={
                        **dict(candidate.get("provenance") or {}),
                        "candidate_id": candidate_id,
                        "write_policy": candidate["policy_version"],
                        "write_reason": "user_confirmed_candidate",
                        "confirmed_by_user": True,
                    },
                )
                resolved = self.store.resolve_memory_candidate(
                    candidate_id,
                    "accepted",
                    memory_id=outcome.memory["memory_id"],
                    resolved_via="memory_candidate_api",
                    resolution_reason="explicit_user_confirmation",
                    accepted_memory_type=outcome.memory["memory_type"],
                    workspace_id=self.workspace_id,
                )
        except Exception:
            if write_attempted:
                self._best_effort_reconcile()
            raise
        return {
            "candidate": resolved,
            "memory": outcome.memory,
            "action": outcome.action,
            "vector_action": outcome.vector_action,
        }

    def reject_candidate(self, candidate_id):
        return self.store.resolve_memory_candidate(
            candidate_id,
            "rejected",
            resolved_via="memory_candidate_api",
            resolution_reason="explicit_user_rejection",
            workspace_id=self.workspace_id,
        )

    def edit(self, memory_id, value, *, source_message_id=None):
        current = self.store.get_memory(
            memory_id, workspace_id=self.workspace_id
        )
        if current["memory_type"] == "collection" or current["slot"] in self.manager.extractor.COLLECTION_SLOTS:
            siblings = self.store.list_memories(
                workspace_id=self.workspace_id,
                slot=current["slot"],
                limit=10_000,
            )
            duplicate = next(
                (
                    item
                    for item in siblings
                    if item["memory_id"] != current["memory_id"]
                    and self._normalize(item["value"])
                    == self._normalize(value)
                ),
                None,
            )
            if duplicate is not None:
                raise ConflictError(
                    "collection memory with the same value already exists"
                )
        provenance = dict(current.get("provenance") or {})
        provenance["reason"] = "explicit_user_edit"
        vector_attempted = False
        try:
            with self.store.transaction():
                updated = self.store.edit_memory(
                    memory_id,
                    value,
                    source="user_edit",
                    provenance=provenance,
                    source_message_id=source_message_id,
                    workspace_id=self.workspace_id,
                )
                vector_attempted = True
                vector = self.manager.edit_memory(
                    memory_id, self._value_text(value)
                )
                if vector is None:
                    vector = self.manager.upsert_memory(
                        provenance.get("semantic_type", "fact"),
                        current["slot"],
                        self._value_text(value),
                        cardinality=current["memory_type"],
                        source="user_edit",
                        source_message_id=source_message_id,
                        memory_id=memory_id,
                    )
        except Exception:
            if vector_attempted:
                self._best_effort_reconcile()
            raise
        return ProductMemoryOutcome("updated", updated, vector.action)

    def delete(self, memory_id):
        """Remove from long-term retrieval; session messages remain unchanged."""
        vector_attempted = False
        try:
            with self.store.transaction():
                deleted = self.store.delete_memory(
                    memory_id,
                    source="user",
                    provenance={"reason": "explicit_user_request"},
                    workspace_id=self.workspace_id,
                )
                vector_attempted = True
                self.manager.delete_memory(memory_id)
        except Exception:
            if vector_attempted:
                self._best_effort_reconcile()
            raise
        return deleted

    def purge(self, memory_id):
        current = self.store.get_memory(
            memory_id,
            workspace_id=self.workspace_id,
            include_deleted=True,
        )
        if current["status"] != "deleted":
            raise ConflictError("memory must be logically deleted before purge")

        # Validation happens before the derived FAISS copy is removed.  A
        # rejected purge must not make an active SQLite memory unrecallable.
        self.manager.purge_memory(memory_id)
        return self.store.purge_memory(
            memory_id,
            workspace_id=self.workspace_id,
            require_deleted=True,
        )

    def forget_from_text(self, text):
        slots = self.manager.extractor.infer_forget_slots(text)
        if not slots:
            return 0
        value = self.manager.extractor.forget_value(text)
        active = self.store.list_memories(
            workspace_id=self.workspace_id,
            limit=10_000,
        )
        selected = [
            item
            for item in active
            if ("*" in slots or item["slot"] in slots)
            and (value is None or self._normalize(item["value"]) == self._normalize(value))
        ]
        for item in selected:
            self.delete(item["memory_id"])
        pending_candidates = self.store.list_memory_candidates(
            workspace_id=self.workspace_id,
            status="pending",
            limit=1_000_000,
        )
        selected_candidates = [
            item
            for item in pending_candidates
            if ("*" in slots or item["slot"] in slots)
            and (value is None or self._normalize(item["value"]) == self._normalize(value))
        ]
        for candidate in selected_candidates:
            self.store.resolve_memory_candidate(
                candidate["candidate_id"],
                "rejected",
                resolved_via="forget_request",
                resolution_reason="explicit_user_forget_request",
                workspace_id=self.workspace_id,
            )
        return len(selected) + len(selected_candidates)

    def list(self, *, status="active", memory_type=None, slot=None, limit=100):
        return self.store.list_memories(
            workspace_id=self.workspace_id,
            status=status,
            memory_type=memory_type,
            slot=slot,
            limit=limit,
        )

    def history(self, memory_id):
        return self.store.get_memory_history(
            memory_id, workspace_id=self.workspace_id
        )

    def export(self):
        return self.store.export_workspace(
            self.workspace_id,
            include_deleted=True,
            include_history=True,
        )

    def active_context(self):
        return "\n".join(
            f"{item['slot']}：{self._value_text(item['value'])}"
            for item in self.list(limit=1_000)
        ) or "无。"

    def initialize_index(self, *, import_legacy=False):
        """Adopt SQLite once; later starts only repair its derived index.

        Legacy vector-only data needs explicit import into a new workspace.
        An initialized or previously used product workspace must never adopt
        orphan vectors as new facts, including after all memories are purged.
        """

        imported = 0
        with self.store.transaction():
            initialized = self.store.memory_index_initialized(self.workspace_id)
            has_history = self.store.has_product_history(self.workspace_id)
            if import_legacy and (initialized or has_history):
                raise ConflictError(
                    "legacy import is only allowed into a new, unused workspace; "
                    "normal startup repairs the index from SQLite"
                )
            if not initialized and not has_history:
                if self.manager.list_active_memory_records(limit=1):
                    if not import_legacy:
                        raise ValueError(
                            "existing memory index has no SQLite authority; "
                            "back up the index and run python main.py "
                            "--import-legacy-memories to explicitly adopt it"
                        )
                    imported = self._import_legacy_active_memories()
            self.store.mark_memory_index_initialized(self.workspace_id)
        # The marker and any imported facts are durable before projection
        # repair. If repair fails, the next start retries from SQLite only.
        return imported, self.reconcile_index()

    def reconcile_index(self):
        """Repair the active FAISS projection from SQLite's current state.

        SQLite remains the source of truth; stale vector-only records are
        physically removed and missing or mismatched active records are
        recreated with their stable product ``memory_id``.
        """

        sql_records = self.list(limit=1_000_000)
        sql_by_id = {item["memory_id"]: item for item in sql_records}
        vector_records = self.manager.list_active_memory_records(limit=1_000_000)
        vector_by_id = {
            item.get("memory_id"): item
            for item in vector_records
            if item.get("memory_id")
        }

        removed = 0
        for memory_id in set(vector_by_id) - set(sql_by_id):
            removed += int(self.manager.purge_memory(memory_id) or 0)

        added = 0
        replaced = 0
        for memory_id, item in sql_by_id.items():
            current = vector_by_id.get(memory_id)
            value_text = self._value_text(item["value"])
            semantic_type = (item.get("provenance") or {}).get(
                "semantic_type", item["slot"].split(".", 1)[0]
            )
            expected = (
                item["slot"],
                item["memory_type"],
                self._normalize(value_text),
            )
            actual = None
            if current:
                actual = (
                    current.get("slot"),
                    current.get("cardinality", "scalar"),
                    self._normalize(current.get("value", "")),
                )
            if actual == expected:
                continue
            if current:
                self.manager.purge_memory(memory_id)
                replaced += 1
            else:
                added += 1
            self.manager.upsert_memory(
                semantic_type,
                item["slot"],
                value_text,
                cardinality=item["memory_type"],
                source=item.get("source", "sqlite_reconcile"),
                source_message_id=item.get("source_message_id"),
                memory_id=memory_id,
            )

        return {
            "active_records": len(sql_records),
            "added": added,
            "replaced": replaced,
            "removed": removed,
        }

    def _import_legacy_active_memories(self):
        """Import within initialize_index's explicit one-time transaction."""

        existing = self.list(limit=1_000_000)
        existing_ids = {item["memory_id"] for item in existing}
        existing_values = {
            (item["slot"], self._normalize(item["value"]))
            for item in existing
        }
        active_scalar_slots = {
            item["slot"]
            for item in existing
            if item["memory_type"] == "scalar"
        }
        imported = 0
        # The snapshot is already in memory. Do not silently import just a
        # page and let reconciliation discard the remaining source records.
        for metadata in self.manager.list_active_memory_records(limit=None):
            memory_id = metadata.get("memory_id") or str(uuid4())
            slot = metadata.get("slot")
            value = metadata.get("value", "")
            cardinality = metadata.get("cardinality", "scalar")
            if (
                memory_id in existing_ids
                or (slot, self._normalize(value)) in existing_values
                or (cardinality == "scalar" and slot in active_scalar_slots)
            ):
                continue
            try:
                self.store.create_memory(
                    slot,
                    value,
                    memory_type=cardinality,
                    source="legacy_import",
                    provenance={
                        "semantic_type": metadata.get("memory_type", "fact"),
                        "original_version": metadata.get("version", 1),
                    },
                    memory_id=memory_id,
                    workspace_id=self.workspace_id,
                )
                imported += 1
                existing_ids.add(memory_id)
                existing_values.add((slot, self._normalize(value)))
                if cardinality == "scalar":
                    active_scalar_slots.add(slot)
            except (ValueError, ConflictError) as exc:
                # Do not mark a partial import complete or discard source data.
                raise ValueError(f"invalid legacy memory {memory_id}: {exc}") from exc
        return imported

    def _validate_candidate_snapshot(self, candidate):
        """Reject a scalar candidate when its target slot changed meanwhile.

        Candidate review is optimistic: the proposal records the current
        scalar memory id/version, then confirmation verifies that snapshot.
        An already-identical value is safe to accept as an idempotent no-op.
        """

        if candidate["memory_type"] != "scalar" or candidate["slot"] in self.manager.extractor.COLLECTION_SLOTS:
            return

        current_items = self.store.list_memories(
            workspace_id=self.workspace_id,
            status="active",
            memory_type="scalar",
            slot=candidate["slot"],
            limit=2,
        )
        current = current_items[0] if current_items else None
        if current is not None and self._normalize(
            current["value"]
        ) == self._normalize(candidate["value"]):
            return

        provenance = dict(candidate.get("provenance") or {})
        expected_id = provenance.get("expected_memory_id")
        expected_version = provenance.get("expected_memory_version")

        if expected_id is None:
            if current is None:
                return
            raise ConflictError(
                "memory slot changed since the candidate was proposed; "
                "review or reject it"
            )

        if current is None:
            raise ConflictError(
                "memory slot changed since the candidate was proposed; "
                "review or reject it"
            )
        try:
            version_matches = current["version"] == int(expected_version)
        except (TypeError, ValueError):
            version_matches = False
        if current["memory_id"] != expected_id or not version_matches:
            raise ConflictError(
                "memory slot changed since the candidate was proposed; "
                "review or reject it"
            )

    def _best_effort_reconcile(self):
        """Repair a possibly mutated projection without masking the failure."""

        try:
            self.reconcile_index()
        except Exception:
            # Startup reconciliation remains the final repair path if the
            # derived index itself is temporarily unavailable.
            pass

    @staticmethod
    def _value_text(value):
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False, sort_keys=True)

    @staticmethod
    def _normalize(value):
        return "".join(
            ProductMemoryService._value_text(value).lower().split()
        ).strip("，。！？,.!?")


__all__ = ["ProductMemoryOutcome", "ProductMemoryService", "NotFoundError"]
