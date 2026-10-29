"""用于会话与离散记忆的轻量 SQLite 持久化层。

本存储层仅依赖 Python 标准库。SQLite 是应用状态的权威来源；
上层引入向量索引后，可依据当前记忆记录重新构建索引。
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from functools import wraps
import json
from pathlib import Path
import sqlite3
from threading import RLock
from typing import Any, Iterator, Mapping
from uuid import uuid4


MEMORY_TYPES = frozenset({"scalar", "collection", "event"})
MEMORY_STATUSES = frozenset({"active", "deleted"})
MEMORY_CANDIDATE_STATUSES = frozenset({"pending", "accepted", "rejected"})
MESSAGE_ROLES = frozenset({"system", "user", "assistant", "tool"})


def _synchronized_read(method):
    """保护共享 SQLite 连接，避免暴露尚未提交的写入。"""

    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._write_lock:
            return method(self, *args, **kwargs)

    return wrapped


class ProductStoreError(RuntimeError):
    """ProductStore 抛出的基础异常。"""


class NotFoundError(ProductStoreError):
    """工作空间所属的资源不存在时抛出的异常。"""


class ConflictError(ProductStoreError):
    """写入将违反应用必须保持的数据约束时抛出的异常。"""


class DeletedMemoryError(ConflictError):
    """操作要求目标记忆处于有效状态时抛出的异常。"""


class ProductStore:
    """持久化工作空间、会话和带版本管理的离散记忆。

    接口中的 workspace_id 默认均为 "local"。这使初版应用保持单用户模式，
    同时在数据结构中明确归属，便于以后扩展服务层时避免破坏性迁移。
    """

    def __init__(self, db_path: str | Path, default_workspace_id: str = "local"):
        self.db_path = str(db_path)
        self.default_workspace_id = self._required_text(
            default_workspace_id, "default_workspace_id"
        )
        if self.db_path != ":memory:":
            Path(self.db_path).expanduser().parent.mkdir(parents=True, exist_ok=True)

        self._connection = sqlite3.connect(
            self.db_path,
            isolation_level=None,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA busy_timeout = 5000")
        self._savepoint_counter = 0
        self._write_lock = RLock()
        try:
            self._initialize_schema()
            self.create_workspace(
                self.default_workspace_id,
                name="Local workspace",
            )
        except Exception:
            self._connection.close()
            raise

    @property
    def connection(self) -> sqlite3.Connection:
        """提供已配置的数据库连接，用于诊断和健康检查。"""

        return self._connection

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "ProductStore":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator["ProductStore"]:
        """在事务中原子执行一组操作，并支持安全嵌套的存储调用。"""

        with self._write_lock:
            if self._connection.in_transaction:
                self._savepoint_counter += 1
                savepoint = f"product_store_{self._savepoint_counter}"
                self._connection.execute(f"SAVEPOINT {savepoint}")
                try:
                    yield self
                except Exception:
                    self._connection.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                    self._connection.execute(f"RELEASE SAVEPOINT {savepoint}")
                    raise
                else:
                    self._connection.execute(f"RELEASE SAVEPOINT {savepoint}")
                return

            self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield self
            except Exception:
                self._connection.rollback()
                raise
            else:
                self._connection.commit()

    # 一次性确立 SQLite 为记忆的权威存储。

    @_synchronized_read
    def memory_index_initialized(self, workspace_id: str | None = None) -> bool:
        return self._connection.execute(
            "SELECT 1 FROM memory_index_state WHERE workspace_id = ?",
            (self._workspace_id(workspace_id),),
        ).fetchone() is not None

    def mark_memory_index_initialized(self, workspace_id: str | None = None) -> None:
        with self.transaction():
            self._connection.execute(
                "INSERT OR IGNORE INTO memory_index_state (workspace_id, initialized_at) VALUES (?, ?)",
                (self._workspace_id(workspace_id), self._now()),
            )

    @_synchronized_read
    def has_product_history(self, workspace_id: str | None = None) -> bool:
        workspace_id = self._workspace_id(workspace_id)
        # 已删除的记忆和已审核的候选仍能证明工作空间已有归属数据。
        return any(
            self._connection.execute(
                f"SELECT 1 FROM {table} WHERE workspace_id = ? LIMIT 1",
                (workspace_id,),
            ).fetchone() is not None
            for table in ("sessions", "memory_items", "memory_candidates")
        )

    # 工作空间与会话持久化

    def create_workspace(self, workspace_id: str, name: str | None = None) -> dict:
        workspace_id = self._required_text(workspace_id, "workspace_id")
        display_name = self._required_text(name or workspace_id, "name")
        created_at = self._now()
        with self.transaction():
            self._connection.execute(
                """
                INSERT INTO workspaces (workspace_id, name, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(workspace_id) DO NOTHING
                """,
                (workspace_id, display_name, created_at),
            )
        return self.get_workspace(workspace_id)

    @_synchronized_read
    def get_workspace(self, workspace_id: str | None = None) -> dict:
        workspace_id = self._workspace_id(workspace_id)
        row = self._connection.execute(
            """
            SELECT workspace_id, name, created_at
            FROM workspaces
            WHERE workspace_id = ?
            """,
            (workspace_id,),
        ).fetchone()
        if row is None:
            raise NotFoundError(f"workspace not found: {workspace_id}")
        return dict(row)

    @_synchronized_read
    def list_workspaces(self) -> list[dict]:
        rows = self._connection.execute(
            """
            SELECT workspace_id, name, created_at
            FROM workspaces
            ORDER BY created_at, workspace_id
            """
        ).fetchall()
        return [dict(row) for row in rows]

    def create_session(
        self,
        title: str | None = None,
        *,
        session_id: str | None = None,
        workspace_id: str | None = None,
    ) -> dict:
        workspace_id = self._workspace_id(workspace_id)
        self.get_workspace(workspace_id)
        session_id = self._required_text(session_id or str(uuid4()), "session_id")
        title = self._optional_text(title, "title")
        now = self._now()
        try:
            with self.transaction():
                self._connection.execute(
                    """
                    INSERT INTO sessions (
                        session_id, workspace_id, title, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (session_id, workspace_id, title, now, now),
                )
        except sqlite3.IntegrityError as exc:
            raise ConflictError(f"session already exists: {session_id}") from exc
        return self.get_session(session_id, workspace_id=workspace_id)

    @_synchronized_read
    def get_session(
        self, session_id: str, *, workspace_id: str | None = None
    ) -> dict:
        session_id = self._required_text(session_id, "session_id")
        workspace_id = self._workspace_id(workspace_id)
        row = self._connection.execute(
            """
            SELECT
                s.session_id,
                s.workspace_id,
                s.title,
                s.created_at,
                s.updated_at,
                COUNT(m.message_id) AS message_count
            FROM sessions AS s
            LEFT JOIN messages AS m ON m.session_id = s.session_id
            WHERE s.session_id = ? AND s.workspace_id = ?
            GROUP BY s.session_id
            """,
            (session_id, workspace_id),
        ).fetchone()
        if row is None:
            raise NotFoundError(f"session not found: {session_id}")
        result = dict(row)
        result["message_count"] = int(result["message_count"])
        return result

    @_synchronized_read
    def list_sessions(
        self, *, workspace_id: str | None = None, limit: int = 100
    ) -> list[dict]:
        workspace_id = self._workspace_id(workspace_id)
        limit = self._positive_limit(limit)
        rows = self._connection.execute(
            """
            SELECT
                s.session_id,
                s.workspace_id,
                s.title,
                s.created_at,
                s.updated_at,
                COUNT(m.message_id) AS message_count
            FROM sessions AS s
            LEFT JOIN messages AS m ON m.session_id = s.session_id
            WHERE s.workspace_id = ?
            GROUP BY s.session_id
            ORDER BY s.updated_at DESC, s.session_id DESC
            LIMIT ?
            """,
            (workspace_id, limit),
        ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["message_count"] = int(item["message_count"])
            result.append(item)
        return result

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        message_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        workspace_id: str | None = None,
    ) -> dict:
        session_id = self._required_text(session_id, "session_id")
        workspace_id = self._workspace_id(workspace_id)
        self.get_session(session_id, workspace_id=workspace_id)
        if role not in MESSAGE_ROLES:
            raise ValueError(f"role must be one of: {', '.join(sorted(MESSAGE_ROLES))}")
        content = self._required_text(content, "content")
        message_id = self._required_text(message_id or str(uuid4()), "message_id")
        metadata_json = self._json_dump(metadata or {}, "metadata")
        created_at = self._now()

        try:
            with self.transaction():
                self._connection.execute(
                    """
                    INSERT INTO messages (
                        message_id, session_id, role, content,
                        metadata_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        message_id,
                        session_id,
                        role,
                        content,
                        metadata_json,
                        created_at,
                    ),
                )
                self._connection.execute(
                    "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                    (created_at, session_id),
                )
        except sqlite3.IntegrityError as exc:
            raise ConflictError(f"message already exists: {message_id}") from exc
        return self.get_message(message_id, workspace_id=workspace_id)

    @_synchronized_read
    def get_message(
        self, message_id: str, *, workspace_id: str | None = None
    ) -> dict:
        message_id = self._required_text(message_id, "message_id")
        workspace_id = self._workspace_id(workspace_id)
        row = self._connection.execute(
            """
            SELECT
                m.message_id,
                m.session_id,
                s.workspace_id,
                m.role,
                m.content,
                m.metadata_json,
                m.created_at
            FROM messages AS m
            JOIN sessions AS s ON s.session_id = m.session_id
            WHERE m.message_id = ? AND s.workspace_id = ?
            """,
            (message_id, workspace_id),
        ).fetchone()
        if row is None:
            raise NotFoundError(f"message not found: {message_id}")
        return self._decode_message(row)

    @_synchronized_read
    def list_messages(
        self,
        session_id: str,
        *,
        workspace_id: str | None = None,
        limit: int = 1000,
    ) -> list[dict]:
        session_id = self._required_text(session_id, "session_id")
        workspace_id = self._workspace_id(workspace_id)
        self.get_session(session_id, workspace_id=workspace_id)
        limit = self._positive_limit(limit)
        rows = self._connection.execute(
            """
            SELECT
                m.message_id,
                m.session_id,
                s.workspace_id,
                m.role,
                m.content,
                m.metadata_json,
                m.created_at
            FROM messages AS m
            JOIN sessions AS s ON s.session_id = m.session_id
            WHERE m.session_id = ? AND s.workspace_id = ?
            ORDER BY m.created_at DESC, m.rowid DESC
            LIMIT ?
            """,
            (session_id, workspace_id, limit),
        ).fetchall()
        rows.reverse()
        return [self._decode_message(row) for row in rows]

    def delete_session(
        self, session_id: str, *, workspace_id: str | None = None
    ) -> bool:
        session_id = self._required_text(session_id, "session_id")
        workspace_id = self._workspace_id(workspace_id)
        with self.transaction():
            cursor = self._connection.execute(
                "DELETE FROM sessions WHERE session_id = ? AND workspace_id = ?",
                (session_id, workspace_id),
            )
        return cursor.rowcount > 0

    # 等待审核的记忆保存候选

    def create_memory_candidate(
        self,
        slot: str,
        value: Any,
        *,
        memory_type: str = "scalar",
        semantic_type: str = "fact",
        reason_code: str,
        policy_version: str,
        provenance: Mapping[str, Any] | None = None,
        session_id: str | None = None,
        source_message_id: str | None = None,
        candidate_id: str | None = None,
        workspace_id: str | None = None,
    ) -> dict:
        workspace_id = self._workspace_id(workspace_id)
        self.get_workspace(workspace_id)
        slot = self._required_text(slot, "slot")
        memory_type = self._memory_type(memory_type)
        semantic_type = self._required_text(semantic_type, "semantic_type")
        reason_code = self._required_text(reason_code, "reason_code")
        policy_version = self._required_text(policy_version, "policy_version")
        candidate_id = self._required_text(
            candidate_id or str(uuid4()), "candidate_id"
        )
        value_json = self._json_dump(value, "value")
        provenance_json = self._json_dump(provenance or {}, "provenance")
        if session_id is not None:
            session_id = self._required_text(session_id, "session_id")
            self.get_session(session_id, workspace_id=workspace_id)
        source_message_id = self._validate_source_message(
            source_message_id, workspace_id
        )
        if source_message_id is not None and session_id is not None:
            source_message = self.get_message(
                source_message_id, workspace_id=workspace_id
            )
            if source_message["session_id"] != session_id:
                raise ValueError("source_message_id must belong to session_id")
        now = self._now()

        try:
            with self.transaction():
                self._connection.execute(
                    """
                    INSERT INTO memory_candidates (
                        candidate_id, workspace_id, session_id,
                        source_message_id, slot, memory_type, semantic_type,
                        value_json, status, reason_code, policy_version,
                        provenance_json, memory_id, created_at, resolved_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?, NULL, ?, NULL)
                    """,
                    (
                        candidate_id,
                        workspace_id,
                        session_id,
                        source_message_id,
                        slot,
                        memory_type,
                        semantic_type,
                        value_json,
                        reason_code,
                        policy_version,
                        provenance_json,
                        now,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise ConflictError(
                f"memory candidate already exists: {candidate_id}"
            ) from exc
        return self.get_memory_candidate(candidate_id, workspace_id=workspace_id)

    @_synchronized_read
    def get_memory_candidate(
        self, candidate_id: str, *, workspace_id: str | None = None
    ) -> dict:
        candidate_id = self._required_text(candidate_id, "candidate_id")
        workspace_id = self._workspace_id(workspace_id)
        row = self._connection.execute(
            """
            SELECT * FROM memory_candidates
            WHERE candidate_id = ? AND workspace_id = ?
            """,
            (candidate_id, workspace_id),
        ).fetchone()
        if row is None:
            raise NotFoundError(f"memory candidate not found: {candidate_id}")
        return self._decode_memory_candidate(row)

    @_synchronized_read
    def list_memory_candidates(
        self,
        *,
        workspace_id: str | None = None,
        status: str | None = "pending",
        limit: int = 100,
    ) -> list[dict]:
        workspace_id = self._workspace_id(workspace_id)
        limit = self._positive_limit(limit)
        clauses = ["workspace_id = ?"]
        params: list[Any] = [workspace_id]
        if status is not None:
            if status not in MEMORY_CANDIDATE_STATUSES:
                raise ValueError(
                    "status must be one of: "
                    + ", ".join(sorted(MEMORY_CANDIDATE_STATUSES))
                )
            clauses.append("status = ?")
            params.append(status)
        params.append(limit)
        rows = self._connection.execute(
            f"""
            SELECT * FROM memory_candidates
            WHERE {' AND '.join(clauses)}
            ORDER BY created_at DESC, candidate_id DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [self._decode_memory_candidate(row) for row in rows]

    def resolve_memory_candidate(
        self,
        candidate_id: str,
        status: str,
        *,
        memory_id: str | None = None,
        resolved_via: str = "local_product",
        resolution_reason: str | None = None,
        accepted_memory_type: str | None = None,
        workspace_id: str | None = None,
    ) -> dict:
        candidate_id = self._required_text(candidate_id, "candidate_id")
        workspace_id = self._workspace_id(workspace_id)
        resolved_via = self._required_text(resolved_via, "resolved_via")
        resolution_reason = self._optional_text(
            resolution_reason, "resolution_reason"
        )
        if status not in {"accepted", "rejected"}:
            raise ValueError("status must be accepted or rejected")
        if status == "accepted":
            memory_id = self._required_text(memory_id, "memory_id")
        elif memory_id is not None:
            raise ValueError("rejected candidates cannot reference a memory")

        with self.transaction():
            candidate = self.get_memory_candidate(
                candidate_id, workspace_id=workspace_id
            )
            if candidate["status"] != "pending":
                raise ConflictError(
                    f"memory candidate is already {candidate['status']}: {candidate_id}"
                )
            if status == "accepted":
                memory = self.get_memory(memory_id, workspace_id=workspace_id)
                # 服务可以升级旧候选的单值或集合类型。
                # 槽位和值仍须完全匹配，并记录该变更以供审计。
                resolved_type = self._memory_type(accepted_memory_type or candidate["memory_type"])
                if (
                    memory["slot"] != candidate["slot"]
                    or memory["memory_type"] != resolved_type
                    or self._json_dump(memory["value"], "memory value")
                    != self._json_dump(candidate["value"], "candidate value")
                ):
                    raise ConflictError(
                        "accepted candidate must match the linked memory"
                    )
            provenance = dict(candidate.get("provenance") or {})
            resolved_type = resolved_type if status == "accepted" else candidate["memory_type"]
            if resolved_type != candidate["memory_type"]:
                provenance["original_memory_type"] = candidate["memory_type"]
            provenance["resolution"] = {
                "status": status,
                "via": resolved_via,
                "reason": resolution_reason,
            }
            self._connection.execute(
                """
                UPDATE memory_candidates
                SET status = ?, memory_id = ?, provenance_json = ?, resolved_at = ?, memory_type = ?
                WHERE candidate_id = ? AND workspace_id = ?
                """,
                (
                    status,
                    memory_id,
                    self._json_dump(provenance, "provenance"),
                    self._now(),
                    resolved_type,
                    candidate_id,
                    workspace_id,
                ),
            )
        return self.get_memory_candidate(candidate_id, workspace_id=workspace_id)

    # 带版本管理的离散记忆生命周期

    def create_memory(
        self,
        slot: str,
        value: Any,
        *,
        memory_type: str = "scalar",
        source: str = "user",
        provenance: Mapping[str, Any] | None = None,
        source_message_id: str | None = None,
        memory_id: str | None = None,
        workspace_id: str | None = None,
    ) -> dict:
        workspace_id = self._workspace_id(workspace_id)
        self.get_workspace(workspace_id)
        slot = self._required_text(slot, "slot")
        memory_type = self._memory_type(memory_type)
        source = self._required_text(source, "source")
        memory_id = self._required_text(memory_id or str(uuid4()), "memory_id")
        value_json = self._json_dump(value, "value")
        provenance_json = self._json_dump(provenance or {}, "provenance")
        source_message_id = self._validate_source_message(
            source_message_id, workspace_id
        )
        now = self._now()

        try:
            with self.transaction():
                self._connection.execute(
                    """
                    INSERT INTO memory_items (
                        memory_id, workspace_id, memory_type, slot, status,
                        current_version, created_at, updated_at, deleted_at
                    ) VALUES (?, ?, ?, ?, 'active', 1, ?, ?, NULL)
                    """,
                    (memory_id, workspace_id, memory_type, slot, now, now),
                )
                self._connection.execute(
                    """
                    INSERT INTO memory_revisions (
                        memory_id, version, operation, status, value_json,
                        source, provenance_json, source_message_id, created_at
                    ) VALUES (?, 1, 'created', 'active', ?, ?, ?, ?, ?)
                    """,
                    (
                        memory_id,
                        value_json,
                        source,
                        provenance_json,
                        source_message_id,
                        now,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            if memory_type == "scalar":
                message = f"active scalar memory already exists for slot: {slot}"
            else:
                message = f"memory already exists: {memory_id}"
            raise ConflictError(message) from exc
        return self.get_memory(memory_id, workspace_id=workspace_id)

    def edit_memory(
        self,
        memory_id: str,
        value: Any,
        *,
        source: str = "user",
        provenance: Mapping[str, Any] | None = None,
        source_message_id: str | None = None,
        workspace_id: str | None = None,
    ) -> dict:
        memory_id = self._required_text(memory_id, "memory_id")
        workspace_id = self._workspace_id(workspace_id)
        source = self._required_text(source, "source")
        value_json = self._json_dump(value, "value")
        provenance_json = self._json_dump(provenance or {}, "provenance")
        source_message_id = self._validate_source_message(
            source_message_id, workspace_id
        )

        with self.transaction():
            item = self._memory_item(memory_id, workspace_id)
            if item["status"] == "deleted":
                raise DeletedMemoryError(f"memory is deleted: {memory_id}")
            version = int(item["current_version"]) + 1
            now = self._now()
            self._connection.execute(
                """
                INSERT INTO memory_revisions (
                    memory_id, version, operation, status, value_json,
                    source, provenance_json, source_message_id, created_at
                ) VALUES (?, ?, 'edited', 'active', ?, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    version,
                    value_json,
                    source,
                    provenance_json,
                    source_message_id,
                    now,
                ),
            )
            self._connection.execute(
                """
                UPDATE memory_items
                SET current_version = ?, updated_at = ?
                WHERE memory_id = ?
                """,
                (version, now, memory_id),
            )
        return self.get_memory(memory_id, workspace_id=workspace_id)

    def delete_memory(
        self,
        memory_id: str,
        *,
        source: str = "user",
        provenance: Mapping[str, Any] | None = None,
        workspace_id: str | None = None,
    ) -> dict:
        """逻辑删除记忆，同时保留可审计的版本记录。"""

        memory_id = self._required_text(memory_id, "memory_id")
        workspace_id = self._workspace_id(workspace_id)
        source = self._required_text(source, "source")
        provenance_json = self._json_dump(provenance or {}, "provenance")

        with self.transaction():
            item = self._memory_item(memory_id, workspace_id)
            if item["status"] == "deleted":
                return self.get_memory(
                    memory_id,
                    workspace_id=workspace_id,
                    include_deleted=True,
                )
            current = self._current_memory_revision(memory_id)
            version = int(item["current_version"]) + 1
            now = self._now()
            self._connection.execute(
                """
                INSERT INTO memory_revisions (
                    memory_id, version, operation, status, value_json,
                    source, provenance_json, source_message_id, created_at
                ) VALUES (?, ?, 'deleted', 'deleted', ?, ?, ?, NULL, ?)
                """,
                (
                    memory_id,
                    version,
                    current["value_json"],
                    source,
                    provenance_json,
                    now,
                ),
            )
            self._connection.execute(
                """
                UPDATE memory_items
                SET status = 'deleted', current_version = ?, updated_at = ?,
                    deleted_at = ?
                WHERE memory_id = ?
                """,
                (version, now, now, memory_id),
            )
        return self.get_memory(
            memory_id,
            workspace_id=workspace_id,
            include_deleted=True,
        )

    def purge_memory(
        self,
        memory_id: str,
        *,
        workspace_id: str | None = None,
        require_deleted: bool = True,
    ) -> bool:
        """物理移除记忆及其全部版本。

        require_deleted 默认为 true，防止界面或接口调用方
        意外绕过可审核的逻辑删除步骤。
        """

        memory_id = self._required_text(memory_id, "memory_id")
        workspace_id = self._workspace_id(workspace_id)
        with self.transaction():
            item = self._memory_item(memory_id, workspace_id)
            if require_deleted and item["status"] != "deleted":
                raise ConflictError("memory must be logically deleted before purge")
            # 已确认候选属于记忆的保存授权历史。
            # 物理清除时须先移除候选中重复保存的值，避免外键操作
            # 仅解除关联，却留下对应的候选内容。
            self._connection.execute(
                """
                DELETE FROM memory_candidates
                WHERE workspace_id = ? AND memory_id = ?
                """,
                (workspace_id, memory_id),
            )
            cursor = self._connection.execute(
                "DELETE FROM memory_items WHERE memory_id = ? AND workspace_id = ?",
                (memory_id, workspace_id),
            )
        return cursor.rowcount > 0

    @_synchronized_read
    def get_memory(
        self,
        memory_id: str,
        *,
        workspace_id: str | None = None,
        include_deleted: bool = False,
    ) -> dict:
        memory_id = self._required_text(memory_id, "memory_id")
        workspace_id = self._workspace_id(workspace_id)
        row = self._connection.execute(
            """
            SELECT
                i.memory_id,
                i.workspace_id,
                i.memory_type,
                i.slot,
                i.status,
                i.current_version AS version,
                i.created_at,
                i.updated_at,
                i.deleted_at,
                r.operation,
                r.value_json,
                r.source,
                r.provenance_json,
                r.source_message_id,
                r.created_at AS revision_created_at
            FROM memory_items AS i
            JOIN memory_revisions AS r
              ON r.memory_id = i.memory_id
             AND r.version = i.current_version
            WHERE i.memory_id = ? AND i.workspace_id = ?
            """,
            (memory_id, workspace_id),
        ).fetchone()
        if row is None or (row["status"] == "deleted" and not include_deleted):
            raise NotFoundError(f"memory not found: {memory_id}")
        return self._decode_memory(row)

    @_synchronized_read
    def list_memories(
        self,
        *,
        workspace_id: str | None = None,
        status: str | None = "active",
        memory_type: str | None = None,
        slot: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """列出当前版本；传入 status=None 时也包含已删除记录。"""

        workspace_id = self._workspace_id(workspace_id)
        limit = self._positive_limit(limit)
        if status is not None and status not in MEMORY_STATUSES:
            raise ValueError(
                f"status must be one of: {', '.join(sorted(MEMORY_STATUSES))}"
            )
        if memory_type is not None:
            memory_type = self._memory_type(memory_type)
        if slot is not None:
            slot = self._required_text(slot, "slot")

        clauses = ["i.workspace_id = ?"]
        params: list[Any] = [workspace_id]
        if status is not None:
            clauses.append("i.status = ?")
            params.append(status)
        if memory_type is not None:
            clauses.append("i.memory_type = ?")
            params.append(memory_type)
        if slot is not None:
            clauses.append("i.slot = ?")
            params.append(slot)
        params.append(limit)

        rows = self._connection.execute(
            f"""
            SELECT
                i.memory_id,
                i.workspace_id,
                i.memory_type,
                i.slot,
                i.status,
                i.current_version AS version,
                i.created_at,
                i.updated_at,
                i.deleted_at,
                r.operation,
                r.value_json,
                r.source,
                r.provenance_json,
                r.source_message_id,
                r.created_at AS revision_created_at
            FROM memory_items AS i
            JOIN memory_revisions AS r
              ON r.memory_id = i.memory_id
             AND r.version = i.current_version
            WHERE {' AND '.join(clauses)}
            ORDER BY i.updated_at DESC, i.memory_id DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [self._decode_memory(row) for row in rows]

    @_synchronized_read
    def get_memory_history(
        self, memory_id: str, *, workspace_id: str | None = None
    ) -> list[dict]:
        memory_id = self._required_text(memory_id, "memory_id")
        workspace_id = self._workspace_id(workspace_id)
        item = self._memory_item(memory_id, workspace_id)
        rows = self._connection.execute(
            """
            SELECT
                i.memory_id,
                i.workspace_id,
                i.memory_type,
                i.slot,
                r.status,
                r.version,
                i.created_at,
                i.updated_at,
                i.deleted_at,
                r.operation,
                r.value_json,
                r.source,
                r.provenance_json,
                r.source_message_id,
                r.created_at AS revision_created_at
            FROM memory_revisions AS r
            JOIN memory_items AS i ON i.memory_id = r.memory_id
            WHERE r.memory_id = ? AND i.workspace_id = ?
            ORDER BY r.version
            """,
            (memory_id, workspace_id),
        ).fetchall()
        if not rows:  # 防御性检查：外键约束正常生效时不应出现此情况。
            raise NotFoundError(f"memory history not found: {item['memory_id']}")
        return [self._decode_memory(row) for row in rows]

    @_synchronized_read
    def export_workspace(
        self,
        workspace_id: str | None = None,
        *,
        include_deleted: bool = True,
        include_history: bool = True,
    ) -> dict:
        """返回可序列化为 JSON、归用户所有的工作空间导出数据。"""

        workspace = self.get_workspace(workspace_id)
        workspace_id = workspace["workspace_id"]
        sessions = self.list_sessions(workspace_id=workspace_id, limit=1_000_000)
        for session in sessions:
            session["messages"] = self.list_messages(
                session["session_id"],
                workspace_id=workspace_id,
                limit=1_000_000,
            )

        memories = self.list_memories(
            workspace_id=workspace_id,
            status=None if include_deleted else "active",
            limit=1_000_000,
        )
        if include_history:
            for memory in memories:
                memory["history"] = self.get_memory_history(
                    memory["memory_id"], workspace_id=workspace_id
                )
        return {
            "schema_version": 2,
            "exported_at": self._now(),
            "workspace": workspace,
            "sessions": sessions,
            "memories": memories,
            "memory_candidates": self.list_memory_candidates(
                workspace_id=workspace_id,
                status=None,
                limit=1_000_000,
            ),
        }

    @_synchronized_read
    def export_workspace_json(
        self,
        workspace_id: str | None = None,
        *,
        include_deleted: bool = True,
        include_history: bool = True,
        indent: int | None = 2,
    ) -> str:
        return json.dumps(
            self.export_workspace(
                workspace_id,
                include_deleted=include_deleted,
                include_history=include_history,
            ),
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
        )

    # 内部辅助方法

    def _initialize_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS workspaces (
                workspace_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                title TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (workspace_id)
                    REFERENCES workspaces(workspace_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_sessions_workspace_updated
                ON sessions(workspace_id, updated_at DESC);

            CREATE TABLE IF NOT EXISTS messages (
                message_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL
                    CHECK (role IN ('system', 'user', 'assistant', 'tool')),
                content TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id)
                    REFERENCES sessions(session_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_messages_session_created
                ON messages(session_id, created_at);

            CREATE TABLE IF NOT EXISTS memory_items (
                memory_id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                memory_type TEXT NOT NULL
                    CHECK (memory_type IN ('scalar', 'collection', 'event')),
                slot TEXT NOT NULL,
                status TEXT NOT NULL
                    CHECK (status IN ('active', 'deleted')),
                current_version INTEGER NOT NULL CHECK (current_version >= 1),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                deleted_at TEXT,
                FOREIGN KEY (workspace_id)
                    REFERENCES workspaces(workspace_id) ON DELETE CASCADE
            );

            CREATE UNIQUE INDEX IF NOT EXISTS uq_active_scalar_slot
                ON memory_items(workspace_id, slot)
                WHERE memory_type = 'scalar' AND status = 'active';

            CREATE INDEX IF NOT EXISTS idx_memory_items_workspace_status
                ON memory_items(workspace_id, status, memory_type, slot);

            CREATE TABLE IF NOT EXISTS memory_revisions (
                memory_id TEXT NOT NULL,
                version INTEGER NOT NULL CHECK (version >= 1),
                operation TEXT NOT NULL
                    CHECK (operation IN ('created', 'edited', 'deleted')),
                status TEXT NOT NULL
                    CHECK (status IN ('active', 'deleted')),
                value_json TEXT NOT NULL,
                source TEXT NOT NULL,
                provenance_json TEXT NOT NULL DEFAULT '{}',
                source_message_id TEXT,
                created_at TEXT NOT NULL,
                PRIMARY KEY (memory_id, version),
                FOREIGN KEY (memory_id)
                    REFERENCES memory_items(memory_id) ON DELETE CASCADE,
                FOREIGN KEY (source_message_id)
                    REFERENCES messages(message_id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS memory_candidates (
                candidate_id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                session_id TEXT,
                source_message_id TEXT,
                slot TEXT NOT NULL,
                memory_type TEXT NOT NULL
                    CHECK (memory_type IN ('scalar', 'collection', 'event')),
                semantic_type TEXT NOT NULL,
                value_json TEXT NOT NULL,
                status TEXT NOT NULL
                    CHECK (status IN ('pending', 'accepted', 'rejected')),
                reason_code TEXT NOT NULL,
                policy_version TEXT NOT NULL,
                provenance_json TEXT NOT NULL DEFAULT '{}',
                memory_id TEXT,
                created_at TEXT NOT NULL,
                resolved_at TEXT,
                FOREIGN KEY (workspace_id)
                    REFERENCES workspaces(workspace_id) ON DELETE CASCADE,
                FOREIGN KEY (session_id)
                    REFERENCES sessions(session_id) ON DELETE SET NULL,
                FOREIGN KEY (source_message_id)
                    REFERENCES messages(message_id) ON DELETE SET NULL,
                FOREIGN KEY (memory_id)
                    REFERENCES memory_items(memory_id) ON DELETE SET NULL
            );

            CREATE INDEX IF NOT EXISTS idx_memory_candidates_workspace_status
                ON memory_candidates(workspace_id, status, created_at DESC);

            CREATE UNIQUE INDEX IF NOT EXISTS uq_pending_memory_candidate
                ON memory_candidates(
                    workspace_id, memory_type, slot, value_json
                )
                WHERE status = 'pending';

            CREATE TABLE IF NOT EXISTS memory_index_state (
                workspace_id TEXT PRIMARY KEY REFERENCES workspaces(workspace_id),
                initialized_at TEXT NOT NULL
            );

            PRAGMA user_version = 3;
            """
        )

    def _memory_item(self, memory_id: str, workspace_id: str) -> sqlite3.Row:
        row = self._connection.execute(
            """
            SELECT * FROM memory_items
            WHERE memory_id = ? AND workspace_id = ?
            """,
            (memory_id, workspace_id),
        ).fetchone()
        if row is None:
            raise NotFoundError(f"memory not found: {memory_id}")
        return row

    def _current_memory_revision(self, memory_id: str) -> sqlite3.Row:
        row = self._connection.execute(
            """
            SELECT r.*
            FROM memory_revisions AS r
            JOIN memory_items AS i
              ON i.memory_id = r.memory_id
             AND i.current_version = r.version
            WHERE r.memory_id = ?
            """,
            (memory_id,),
        ).fetchone()
        if row is None:
            raise NotFoundError(f"memory revision not found: {memory_id}")
        return row

    def _validate_source_message(
        self, source_message_id: str | None, workspace_id: str
    ) -> str | None:
        if source_message_id is None:
            return None
        source_message_id = self._required_text(
            source_message_id, "source_message_id"
        )
        self.get_message(source_message_id, workspace_id=workspace_id)
        return source_message_id

    @staticmethod
    def _decode_message(row: sqlite3.Row) -> dict:
        result = dict(row)
        result["metadata"] = json.loads(result.pop("metadata_json"))
        return result

    @staticmethod
    def _decode_memory(row: sqlite3.Row) -> dict:
        result = dict(row)
        result["version"] = int(result["version"])
        result["value"] = json.loads(result.pop("value_json"))
        result["provenance"] = json.loads(result.pop("provenance_json"))
        return result

    @staticmethod
    def _decode_memory_candidate(row: sqlite3.Row) -> dict:
        result = dict(row)
        result["value"] = json.loads(result.pop("value_json"))
        result["provenance"] = json.loads(result.pop("provenance_json"))
        return result

    def _workspace_id(self, workspace_id: str | None) -> str:
        return self._required_text(
            workspace_id or self.default_workspace_id, "workspace_id"
        )

    @staticmethod
    def _memory_type(memory_type: str) -> str:
        if memory_type not in MEMORY_TYPES:
            raise ValueError(
                f"memory_type must be one of: {', '.join(sorted(MEMORY_TYPES))}"
            )
        return memory_type

    @staticmethod
    def _positive_limit(limit: int) -> int:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")
        return limit

    @staticmethod
    def _required_text(value: Any, field: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a non-empty string")
        return value.strip()

    @staticmethod
    def _optional_text(value: str | None, field: str) -> str | None:
        if value is None:
            return None
        return ProductStore._required_text(value, field)

    @staticmethod
    def _json_dump(value: Any, field: str) -> str:
        try:
            return json.dumps(
                value,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be JSON-serializable") from exc

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="microseconds")
