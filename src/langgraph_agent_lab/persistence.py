"""Checkpointer adapter."""

from __future__ import annotations

import os


def build_checkpointer(
    kind: str = "memory", database_url: str | None = None,
) -> object | None:
    """Return a LangGraph checkpointer.

    Supports: "none", "memory", "sqlite".
    SQLite uses WAL mode for concurrent read safety.
    """
    if kind == "none":
        return None
    if kind == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()
    if kind == "sqlite":
        import sqlite3

        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError:
            try:
                from langgraph_checkpoint_sqlite import SqliteSaver  # type: ignore[import-untyped]
            except ImportError as exc:
                raise RuntimeError(
                    "Install: pip install langgraph-checkpoint-sqlite"
                ) from exc

        db_path = database_url or os.getenv("SQLITE_DB_PATH", "outputs/checkpoints.db")
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        return SqliteSaver(conn=conn)
    if kind == "postgres":
        raise NotImplementedError(
            "Postgres checkpointer is an optional extension. "
            "Use SQLite for persistence evidence."
        )
    raise ValueError(f"Unknown checkpointer kind: {kind}")
