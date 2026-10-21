"""Thin product orchestration for sessions, memory, documents and job fit.

All components are injected.  Constructing :class:`ProductApplication` never
loads an embedding model, an LLM, FAISS, or a document from disk.  A later CLI
or HTTP adapter owns that composition and can map the original domain
exceptions without this layer hiding them.
"""

from __future__ import annotations

import json
from os import PathLike
from threading import RLock
from typing import Any, Mapping, Sequence

from product.write_policy import ConsentAwareWritePolicy
from models.llm import LLMUnavailableError


class ProductApplication:
    """Coordinate the local-first product's existing bounded services."""

    def __init__(
        self,
        *,
        store,
        memory_service,
        engine,
        document_lifecycle,
        job_fit_service,
        workspace_id: str | None = None,
        write_policy=None,
        fact_extractor=None,
    ):
        self.store = store
        self.memory = memory_service
        self.engine = engine
        self.documents = document_lifecycle
        self.job_fit_service = job_fit_service
        self.workspace_id = str(
            workspace_id
            or getattr(memory_service, "workspace_id", None)
            or getattr(store, "default_workspace_id", "local")
        ).strip()
        if not self.workspace_id:
            raise ValueError("workspace_id must not be empty")
        memory_workspace = getattr(memory_service, "workspace_id", self.workspace_id)
        if memory_workspace != self.workspace_id:
            raise ValueError("memory service and application workspace must match")
        memory_store = getattr(memory_service, "store", store)
        if memory_store is not store:
            raise ValueError("memory service and application must share ProductStore")

        self.manager = getattr(memory_service, "manager", None)
        if self.manager is None:
            raise TypeError("memory_service must expose its memory manager")
        engine_manager = self._resolve_engine_manager(engine)
        if engine_manager is not None and engine_manager is not self.manager:
            raise ValueError("engine and memory service must share MemoryManager")

        self.query_policy = self._resolve_query_policy(engine)
        self.fact_extractor = fact_extractor or self.manager.extractor
        self.write_policy = write_policy or ConsentAwareWritePolicy()
        if not callable(getattr(self.write_policy, "decide", None)):
            raise TypeError("write_policy must expose decide(text, candidate_count=...)")
        self.store.create_workspace(self.workspace_id)
        # A single mutable short-term buffer backs the current local engine.
        # Serializing stateful operations prevents two HTTP requests from
        # restoring different sessions into it at the same time.
        self._state_lock = RLock()

    # Sessions and chat

    def create_session(
        self,
        title: str | None = None,
        *,
        session_id: str | None = None,
    ) -> dict:
        return self.store.create_session(
            title,
            session_id=session_id,
            workspace_id=self.workspace_id,
        )

    def list_sessions(self, *, limit: int = 100) -> list[dict]:
        return self.store.list_sessions(
            workspace_id=self.workspace_id,
            limit=limit,
        )

    def list_messages(self, session_id: str, *, limit: int = 1_000) -> list[dict]:
        return self.store.list_messages(
            session_id,
            workspace_id=self.workspace_id,
            limit=limit,
        )

    def delete_session(self, session_id: str) -> dict[str, Any]:
        """Delete a conversation and its messages, preserving long-term memory."""
        session_id = self._required_text(session_id, "session_id")
        # Wait for any in-flight chat before deleting its persisted messages.
        with self._state_lock:
            deleted = self.store.delete_session(
                session_id, workspace_id=self.workspace_id
            )
            if deleted:
                # The next chat restores its own session from SQLite.
                self.manager.short_term_history = []
            return {"session_id": session_id, "deleted": deleted}

    def chat(self, session_id: str, message: str) -> dict[str, Any]:
        """Run one turn and persist both messages plus the execution trace.

        Memory admission reads the original user message independently of the
        answer route. Mixed assertion/question turns can answer and stage a
        candidate together; all persistent effects remain governed by the
        memory service and the existing consent policy.
        """

        message = self._required_text(message, "message")
        with self._state_lock:
            self.store.get_session(session_id, workspace_id=self.workspace_id)
            self._restore_short_term(session_id)
            decision = self._decide(message)
            write_decision = None
            extracted = [] if decision.intent == "memory_forget" else self.fact_extractor.extract(message)
            extraction_trace = (None if decision.intent == "memory_forget" else
                                dict(getattr(self.fact_extractor, "last_trace", {}) or {"method": "rule"}))
            if extracted:
                write_decision = self.write_policy.decide(
                    message,
                    candidate_count=len(extracted),
                )
            user_metadata = {
                "intent": decision.intent,
                "route_intent": decision.intent,
                "route_reason": str(getattr(decision, "reason", "")),
                # Session-local evidence; this does not authorize long-term storage.
                "facts": [{"slot": fact.slot, "value": fact.value, "evidence": fact.evidence}
                          for fact in extracted],
            }
            if write_decision is not None:
                user_metadata["write_decision"] = write_decision.to_dict()
            user_message = self.store.add_message(
                session_id,
                "user",
                message,
                metadata=user_metadata,
                workspace_id=self.workspace_id,
            )

            memory_answer, memory_trace = None, None
            if write_decision is not None:
                if write_decision.action == "commit":
                    memory_answer, memory_trace = self._write_from_chat(
                        message,
                        session_id=session_id,
                        source_message_id=user_message["message_id"],
                        reason=getattr(decision, "reason", ""),
                        write_decision=write_decision,
                        facts=extracted,
                    )
                elif write_decision.action == "propose":
                    memory_answer, memory_trace = self._propose_from_chat(
                        message,
                        session_id=session_id,
                        source_message_id=user_message["message_id"],
                        reason=getattr(decision, "reason", ""),
                        write_decision=write_decision,
                        facts=extracted,
                    )
                else:
                    memory_answer, memory_trace = self._ignore_write_from_chat(
                        message,
                        reason=getattr(decision, "reason", ""),
                        write_decision=write_decision,
                    )
            if decision.intent == "memory_forget":
                answer, trace = self._forget_from_chat(
                    message,
                    reason=getattr(decision, "reason", ""),
                )
                self.manager.add_to_short_term(message, answer)
            else:
                execute = getattr(self.engine, "execute", None)
                answer_failed = False
                try:
                    answer = execute(message, decision, user_facts=user_metadata["facts"]) if callable(execute) else self.engine.run(message)
                    answer = self._required_text(answer, "engine answer")
                except LLMUnavailableError:
                    # Memory admission has already completed independently.
                    # Persist a visible outcome rather than hiding it behind a 503.
                    answer_failed = True
                    answer = "回答生成失败：本地模型暂时不可用。请检查 Ollama 后重试提问。"
                    self.manager.add_to_short_term(message, answer)
                trace = self._read_trace(decision)
                if answer_failed:
                    has_effect = memory_trace is not None and bool(
                        memory_trace.get("memory_writes") or memory_trace.get("candidates_created")
                    )
                    trace["status"] = "partial" if has_effect else "failed"
                    trace["answer_error"] = "llm_unavailable"
                if memory_trace is not None:
                    trace["memory_effect"] = memory_trace
                    trace["steps"] += memory_trace["steps"]
                    for key in ("memory_writes", "memory_skipped", "candidates_created", "candidate_ids", "write_decision"):
                        if key in memory_trace:
                            trace[key] = memory_trace[key]
            if extraction_trace is not None:
                trace["extraction"] = extraction_trace
                if extraction_trace.get("status") in {"failed", "filtered"}:
                    warning = "本轮记忆提取未完成或部分内容未通过校验，请查看候选列表；聊天可以继续。"
                    memory_answer = f"{memory_answer}\n{warning}" if memory_answer else warning
                    if trace.get("status") == "completed":
                        trace["status"] = "partial"

            assistant_message = self.store.add_message(
                session_id,
                "assistant",
                answer,
                metadata={"trace": trace, "memory_notice": memory_answer},
                workspace_id=self.workspace_id,
            )
            return {
                "session_id": session_id,
                "answer": answer,
                "user_message": user_message,
                "assistant_message": assistant_message,
                "trace": trace,
            }

    # Governed memory API

    def create_memory(
        self,
        slot: str,
        value: Any,
        *,
        cardinality: str = "scalar",
        semantic_type: str | None = None,
        source_message_id: str | None = None,
        session_id: str | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        audited_provenance = {
            **dict(provenance or {}),
            "write_policy": "explicit-memory-api-v1",
            "write_reason": "explicit_memory_api",
            "explicit_consent": True,
        }
        with self._state_lock:
            outcome = self.memory.remember(
                slot,
                value,
                cardinality=cardinality,
                semantic_type=semantic_type,
                source="user",
                source_message_id=source_message_id,
                session_id=session_id,
                provenance=audited_provenance,
            )
        return self._memory_outcome(outcome)

    def get_memory(self, memory_id: str, *, include_deleted: bool = False) -> dict:
        return self.store.get_memory(
            memory_id,
            workspace_id=self.workspace_id,
            include_deleted=include_deleted,
        )

    def list_memories(
        self,
        *,
        status: str = "active",
        memory_type: str | None = None,
        slot: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        return self.memory.list(
            status=status,
            memory_type=memory_type,
            slot=slot,
            limit=limit,
        )

    def update_memory(
        self,
        memory_id: str,
        value: Any,
        *,
        source_message_id: str | None = None,
    ) -> dict[str, Any]:
        with self._state_lock:
            outcome = self.memory.edit(
                memory_id,
                value,
                source_message_id=source_message_id,
            )
        return self._memory_outcome(outcome)

    def delete_memory(self, memory_id: str) -> dict:
        with self._state_lock:
            return self.memory.delete(memory_id)

    def purge_memory(self, memory_id: str) -> bool:
        with self._state_lock:
            return bool(self.memory.purge(memory_id))

    def memory_history(self, memory_id: str) -> list[dict]:
        return self.memory.history(memory_id)

    def list_memory_candidates(
        self, *, status: str | None = "pending", limit: int = 100
    ) -> list[dict]:
        return self.memory.list_candidates(status=status, limit=limit)

    def confirm_memory_candidate(self, candidate_id: str) -> dict[str, Any]:
        with self._state_lock:
            return self.memory.confirm_candidate(candidate_id)

    def reject_memory_candidate(self, candidate_id: str) -> dict:
        with self._state_lock:
            return self.memory.reject_candidate(candidate_id)

    def export_workspace(self) -> dict:
        return self.memory.export()

    # Knowledge document lifecycle

    def import_document(
        self,
        path: str | PathLike[str],
        *,
        source_id: str | None = None,
        title: str | None = None,
    ) -> dict[str, Any]:
        """Import an API-managed/local path and make its chunks searchable."""

        with self._state_lock:
            change = self.documents.import_document(
                path,
                source_id=source_id,
                title=title,
            )
            sync = self.documents.sync()
            self._bind_knowledge_store(sync)
        return {
            "action": change.action,
            "duplicate_of": change.duplicate_of,
            "document": self._document_payload(change.document),
            "sync": self._sync_payload(sync),
        }

    def list_documents(self) -> list[dict[str, Any]]:
        return [
            self._document_payload(document)
            for document in self.documents.list_documents()
        ]

    def delete_document(
        self,
        *,
        source_id: str | None = None,
        document_id: str | None = None,
    ) -> dict[str, Any]:
        with self._state_lock:
            deleted = self.documents.delete_document(
                source_id=source_id,
                document_id=document_id,
            )
            sync = None
            if deleted:
                sync = self.documents.sync()
                self._bind_knowledge_store(sync)
        return {
            "deleted": bool(deleted),
            "sync": self._sync_payload(sync) if sync is not None else None,
        }

    # Product task workflow

    def job_fit(
        self,
        jd_text: str,
        *,
        source_ids: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Analyze a JD against selected catalog documents and active memory."""

        with self._state_lock:
            records = self._select_documents(source_ids)
            document_context = [
                {
                    "source_id": f"doc:{record.document_id}",
                    "title": record.title,
                    "kind": self._document_kind(record),
                    "content": "\n".join(record.chunks),
                }
                for record in records
            ]
            memory_context = [
                {
                    "memory_id": item["memory_id"],
                    "slot": item["slot"],
                    "value": self._value_text(item["value"]),
                    "status": item["status"],
                }
                for item in self.memory.list(limit=1_000)
            ]
            result = self.job_fit_service.analyze(
                jd_text,
                documents=document_context,
                active_memory=memory_context,
            )

        if hasattr(result, "to_dict"):
            payload = result.to_dict()
        elif isinstance(result, dict):
            payload = dict(result)
        else:
            raise TypeError("job_fit_service.analyze must return a dict-like result")
        payload["sources"] = [
            {
                "source_id": f"doc:{record.document_id}",
                "document_id": record.document_id,
                "catalog_source_id": record.source_id,
                "title": record.title,
                "path": record.path,
            }
            for record in records
        ]
        return payload

    # Internal adapters

    def _restore_short_term(self, session_id: str) -> None:
        messages = self.store.list_messages(
            session_id,
            workspace_id=self.workspace_id,
            limit=1_000,
        )
        turns: list[dict] = []
        pending_user: dict | None = None
        for message in messages:
            if message["role"] == "user":
                pending_user = message
            elif message["role"] == "assistant" and pending_user is not None:
                turns.append({"q": pending_user["content"], "a": message["content"],
                              "facts": (pending_user.get("metadata") or {}).get("facts", [])})
                pending_user = None
        limit = max(1, int(getattr(self.manager, "short_term_limit", 5)))
        self.manager.short_term_history = turns[-limit:]

    def _write_from_chat(
        self,
        message: str,
        *,
        session_id: str,
        source_message_id: str,
        reason: str,
        write_decision,
        facts=None,
    ) -> tuple[str, dict[str, Any]]:
        outcomes = self.memory.remember_extracted(
            message,
            facts=facts,
            source_message_id=source_message_id,
            session_id=session_id,
            source="user_explicit",
            provenance={
                "write_policy": write_decision.policy_version,
                "write_reason": write_decision.reason_code,
                "explicit_consent": write_decision.explicit_consent,
            },
        )
        changed = sum(item.action in {"created", "updated"} for item in outcomes)
        skipped = sum(item.action == "skipped" for item in outcomes)
        if changed:
            answer = f"已记录 {changed} 条长期记忆。"
        elif skipped:
            answer = "这条信息已经记录过了，没有重复写入。"
        else:
            answer = "这句话没有形成可保存的明确长期记忆。"
        return answer, {
            "tool": "save_memory",
            "intent": "memory_write",
            "reason": str(reason),
            "router": "deterministic",
            "status": "completed",
            "steps": 1,
            "memory_writes": changed,
            "memory_skipped": skipped,
            "write_decision": write_decision.to_dict(),
        }

    def _propose_from_chat(
        self,
        message: str,
        *,
        session_id: str,
        source_message_id: str,
        reason: str,
        write_decision,
        facts=None,
    ) -> tuple[str, dict[str, Any]]:
        candidates = self.memory.propose_extracted(
            message,
            facts=facts,
            source_message_id=source_message_id,
            session_id=session_id,
            reason_code=write_decision.reason_code,
            policy_version=write_decision.policy_version,
        )
        if candidates:
            answer = (
                f"有 {len(candidates)} 条信息可以长期记住，"
                "你可以在“记忆治理”中确认或忽略。"
            )
        else:
            answer = "没有新增候选；相同内容可能已在记忆或待确认列表中。"
        return answer, {
            "tool": "propose_memory",
            "intent": "memory_candidate",
            "reason": str(reason),
            "router": "deterministic",
            "status": "completed",
            "steps": 1,
            "memory_writes": 0,
            "candidates_created": len(candidates),
            "candidate_ids": [item["candidate_id"] for item in candidates],
            "write_decision": write_decision.to_dict(),
        }

    def _ignore_write_from_chat(
        self,
        message: str,
        *,
        reason: str,
        write_decision,
    ) -> tuple[str, dict[str, Any]]:
        answer = (
            "已按你的要求不保存这条信息。"
            if write_decision.reason_code == "explicit_opt_out"
            else "这句话没有形成可保存的明确长期记忆。"
        )
        return answer, {
            "tool": "skip_memory",
            "intent": "memory_write_ignored",
            "reason": str(reason),
            "router": "deterministic",
            "status": "completed",
            "steps": 1,
            "memory_writes": 0,
            "candidates_created": 0,
            "write_decision": write_decision.to_dict(),
        }

    def _forget_from_chat(
        self, message: str, *, reason: str
    ) -> tuple[str, dict[str, Any]]:
        handled = self.memory.forget_from_text(message)
        answer = (
            f"已按你的要求处理 {handled} 条长期记忆或待确认候选；聊天历史仍保留。"
            if handled
            else "没有找到可处理的对应记忆或待确认候选。"
        )
        return answer, {
            "tool": "forget_memory",
            "intent": "memory_forget",
            "reason": str(reason),
            "router": "deterministic",
            "status": "completed",
            "steps": 1,
            "memory_records_handled": handled,
        }

    def _read_trace(self, decision) -> dict[str, Any]:
        engine_trace = getattr(self.engine, "last_trace", None)
        if isinstance(engine_trace, Mapping):
            trace = dict(engine_trace)
        else:
            executed = getattr(self.engine, "last_route_decision", None)
            trace = {
                "intent": getattr(executed, "intent", decision.intent),
                "reason": str(
                    getattr(executed, "reason", getattr(decision, "reason", ""))
                ),
                "status": "completed",
                "steps": 1,
            }
            retrieval = getattr(self.engine, "last_retrieval_mode", None)
            if retrieval:
                trace["retrieval_mode"] = retrieval
        trace.setdefault("intent", decision.intent)
        trace.setdefault("status", "completed")
        trace.setdefault("steps", 1)
        trace["application_route"] = decision.intent
        answer_mode = getattr(self.engine, "last_answer_mode", None)
        if answer_mode:
            trace["answer_mode"] = answer_mode
        context_policy = getattr(self.engine, "last_context_policy", None)
        if context_policy:
            trace["context_policy"] = dict(context_policy)
        return trace

    def _decide(self, message: str):
        decision = self.query_policy.decide(message)
        if not isinstance(getattr(decision, "intent", None), str):
            raise TypeError("query policy must return a decision with intent")
        return decision

    def _bind_knowledge_store(self, sync_result) -> None:
        vector_store = getattr(sync_result, "vector_store", None)
        if vector_store is not None:
            self.manager.base_vs = vector_store

    def _select_documents(self, source_ids: Sequence[str] | None):
        records = list(self.documents.list_documents())
        if source_ids is None:
            return records
        if isinstance(source_ids, (str, bytes)):
            raise ValueError("source_ids must be a sequence of identifiers")
        wanted = {self._required_text(value, "source_id") for value in source_ids}
        selected = [
            record
            for record in records
            if record.source_id in wanted
            or record.document_id in wanted
            or f"doc:{record.document_id}" in wanted
        ]
        found = {
            identifier
            for record in selected
            for identifier in (
                record.source_id,
                record.document_id,
                f"doc:{record.document_id}",
            )
        }
        missing = sorted(wanted - found)
        if missing:
            raise ValueError("unknown document source: " + ", ".join(missing))
        return selected

    @staticmethod
    def _resolve_engine_manager(engine):
        if hasattr(engine, "pipeline"):
            return getattr(engine.pipeline, "mm", None)
        return getattr(engine, "mm", None)

    @staticmethod
    def _resolve_query_policy(engine):
        if hasattr(engine, "pipeline"):
            policy = getattr(engine.pipeline, "query_policy", None)
        else:
            policy = getattr(engine, "query_policy", None)
        if policy is None or not callable(getattr(policy, "decide", None)):
            raise TypeError("engine must expose a query policy with decide(query)")
        return policy

    @staticmethod
    def _memory_outcome(outcome) -> dict[str, Any]:
        return {
            "action": outcome.action,
            "memory": outcome.memory,
            "vector_action": outcome.vector_action,
        }

    @staticmethod
    def _document_payload(document) -> dict[str, Any]:
        return {
            "source_id": document.source_id,
            "document_id": document.document_id,
            "title": document.title,
            "path": document.path,
            "checksum": document.checksum,
            "chunks": list(document.chunks),
        }

    @staticmethod
    def _sync_payload(sync) -> dict[str, Any]:
        return {
            "desired_count": sync.desired_count,
            "added": sync.added,
            "metadata_updated": sync.metadata_updated,
            "replaced_records": sync.replaced_records,
            "rebuilt": sync.rebuilt,
        }

    @staticmethod
    def _document_kind(record) -> str:
        label = f"{record.source_id} {record.title}".casefold()
        if "resume" in label or "简历" in label:
            return "resume"
        if "project" in label or "项目" in label:
            return "project"
        return "document"

    @staticmethod
    def _value_text(value: Any) -> str:
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False, sort_keys=True)

    @staticmethod
    def _required_text(value: Any, field: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a non-empty string")
        return value.strip()


__all__ = ["ProductApplication"]
