import os
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Tuple

from langgraph.checkpoint.sqlite import SqliteSaver

_DEFAULT_DIR = "data/checkpoints"
_DEFAULT_DB = "threads.db"


def get_db_path(checkpoint_dir: str = _DEFAULT_DIR, db_name: str = _DEFAULT_DB) -> str:
    """Return the SQLite DB path, creating the directory if needed."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    return os.path.join(checkpoint_dir, db_name)


def get_sqlite_checkpointer(checkpoint_dir: str = _DEFAULT_DIR, db_name: str = _DEFAULT_DB) -> SqliteSaver:
    """
    Returns a synchronous SqliteSaver.
    Use this for CLI, direct SQL ops (reset, health), and TTL cleanup.
    Use AsyncSqliteSaver (via get_db_path) for async graph invocations.
    """
    db_path = get_db_path(checkpoint_dir, db_name)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    return SqliteSaver(conn)


def delete_old_threads(checkpointer: SqliteSaver, ttl_hours: int) -> int:
    """Delete all threads whose last checkpoint is older than ttl_hours.

    Args:
        checkpointer: SqliteSaver instance
        ttl_hours: Time-to-live in hours. Threads not accessed in this period are deleted.

    Returns:
        Number of threads deleted
    """
    cutoff_time = time.time() - (ttl_hours * 3600)

    try:
        cursor = checkpointer.conn.cursor()

        # Find thread_ids with no checkpoints newer than cutoff_time
        cursor.execute("""
            SELECT thread_id FROM checkpoints
            GROUP BY thread_id
            HAVING MAX(ts) < ?
        """, (cutoff_time,))

        old_threads = [row[0] for row in cursor.fetchall()]

        if old_threads:
            placeholders = ",".join("?" * len(old_threads))
            cursor.execute(f"""
                DELETE FROM checkpoints
                WHERE thread_id IN ({placeholders})
            """, old_threads)
            checkpointer.conn.commit()

        return len(old_threads)
    except Exception as e:
        raise RuntimeError(f"Failed to delete old threads: {e}")


def get_db_stats(checkpointer: SqliteSaver) -> dict:
    """Get database statistics: size in bytes and active thread count.

    Args:
        checkpointer: SqliteSaver instance

    Returns:
        Dictionary with 'size_bytes' and 'active_thread_count'
    """
    try:
        cursor = checkpointer.conn.cursor()

        # Get database file size
        db_path = checkpointer.conn.execute(
            "PRAGMA database_list").fetchone()[2]
        db_size_bytes = os.path.getsize(
            db_path) if db_path and os.path.exists(db_path) else 0

        # Count unique thread_ids
        cursor.execute("SELECT COUNT(DISTINCT thread_id) FROM checkpoints")
        active_threads = cursor.fetchone()[0]

        # Count total messages across all threads
        cursor.execute("SELECT COUNT(*) FROM checkpoints")
        total_checkpoints = cursor.fetchone()[0]

        return {
            "size_bytes": db_size_bytes,
            "active_thread_count": active_threads,
            "total_checkpoints": total_checkpoints,
        }
    except Exception as e:
        raise RuntimeError(f"Failed to get database stats: {e}")
