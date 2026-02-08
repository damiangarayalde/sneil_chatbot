import os
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver


def get_sqlite_checkpointer(checkpoint_dir: str = "data/checkpoints", db_name: str = "threads.db") -> SqliteSaver:
    """
    Initializes the database directory and returns a persistent SqliteSaver.
    """
    # Ensure the directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Define the full path to the database file
    db_path = os.path.join(checkpoint_dir, db_name)

    # Create the persistent connection
    # check_same_thread=False is essential for async/web-based chatbot environments
    conn = sqlite3.connect(db_path, check_same_thread=False)

    return SqliteSaver(conn)
