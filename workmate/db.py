"""SQLite persistence for conversations, tasks, and execution logs."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


def now_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class WorkMateDatabase:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=20)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(id)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_messages_conversation
                    ON messages(conversation_id, id);

                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT,
                    title TEXT NOT NULL,
                    user_task TEXT NOT NULL,
                    intent TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    plan_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    report_markdown TEXT,
                    report_path TEXT,
                    chart_path TEXT,
                    csv_path TEXT,
                    created_at TEXT NOT NULL,
                    finished_at TEXT,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(id)
                        ON DELETE SET NULL
                );

                CREATE TABLE IF NOT EXISTS task_steps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    step_no INTEGER NOT NULL,
                    tool_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    detail TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(task_id) REFERENCES tasks(id)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_task_steps_task
                    ON task_steps(task_id, step_no);
                """
            )

    def create_conversation(self, title: str = "新对话") -> str:
        conversation_id = str(uuid.uuid4())
        timestamp = now_text()
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO conversations(id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (conversation_id, title[:60], timestamp, timestamp),
            )
        return conversation_id

    def ensure_conversation(self, conversation_id: str | None) -> str:
        if conversation_id and self.get_conversation(conversation_id):
            return conversation_id
        return self.create_conversation()

    def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
        return dict(row) if row else None

    def list_conversations(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT c.*,
                       COUNT(m.id) AS message_count
                FROM conversations c
                LEFT JOIN messages m ON m.conversation_id = c.id
                GROUP BY c.id
                ORDER BY c.updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def add_message(
        self, conversation_id: str, role: str, content: str
    ) -> None:
        timestamp = now_text()
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO messages(conversation_id, role, content, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (conversation_id, role, content, timestamp),
            )
            conversation = connection.execute(
                "SELECT title FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            title = conversation["title"] if conversation else "对话"
            if role == "user" and title in {"新对话", "对话"}:
                title = content.strip().replace("\n", " ")[:32] or "新对话"
            connection.execute(
                """
                UPDATE conversations
                SET title = ?, updated_at = ?
                WHERE id = ?
                """,
                (title, timestamp, conversation_id),
            )

    def get_messages(
        self, conversation_id: str, limit: int = 100
    ) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM (
                    SELECT * FROM messages
                    WHERE conversation_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                )
                ORDER BY id ASC
                """,
                (conversation_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def create_task(
        self,
        *,
        conversation_id: str | None,
        title: str,
        user_task: str,
        intent: str,
        confidence: float,
        plan_json: str,
    ) -> str:
        task_id = str(uuid.uuid4())
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO tasks(
                    id, conversation_id, title, user_task, intent,
                    confidence, plan_json, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    conversation_id,
                    title[:100],
                    user_task,
                    intent,
                    float(confidence),
                    plan_json,
                    "running",
                    now_text(),
                ),
            )
        return task_id

    def add_task_step(
        self,
        task_id: str,
        step_no: int,
        tool_name: str,
        status: str,
        detail: str = "",
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO task_steps(
                    task_id, step_no, tool_name, status, detail, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (task_id, step_no, tool_name, status, detail, now_text()),
            )

    def finish_task(
        self,
        task_id: str,
        *,
        status: str,
        report_markdown: str = "",
        report_path: str = "",
        chart_path: str = "",
        csv_path: str = "",
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE tasks
                SET status = ?, report_markdown = ?, report_path = ?,
                    chart_path = ?, csv_path = ?, finished_at = ?
                WHERE id = ?
                """,
                (
                    status,
                    report_markdown,
                    report_path,
                    chart_path,
                    csv_path,
                    now_text(),
                    task_id,
                ),
            )

    def list_tasks(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT t.*, c.title AS conversation_title
                FROM tasks t
                LEFT JOIN conversations c ON c.id = t.conversation_id
                ORDER BY t.created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        tasks = []
        for row in rows:
            item = dict(row)
            try:
                item["plan"] = json.loads(item.get("plan_json") or "{}")
            except json.JSONDecodeError:
                item["plan"] = {}
            tasks.append(item)
        return tasks

    def get_task_steps(self, task_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM task_steps
                WHERE task_id = ?
                ORDER BY id ASC
                """,
                (task_id,),
            ).fetchall()
        return [dict(row) for row in rows]
