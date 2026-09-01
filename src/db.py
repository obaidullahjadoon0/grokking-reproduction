"""
Tiny SQLite helper. This is the "SQL" piece of the project — every training
run and every logged metric is persisted here so you can query/compare runs
later without any cloud dashboard.
"""

import sqlite3
import json
import time
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    run_name TEXT,
    config_json TEXT,
    created_at REAL
);

CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    epoch INTEGER,
    train_loss REAL,
    train_acc REAL,
    val_loss REAL,
    val_acc REAL,
    FOREIGN KEY(run_id) REFERENCES runs(run_id)
);
"""


def get_conn(db_path: str):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    return conn


def create_run(db_path: str, run_id: str, run_name: str, config: dict):
    conn = get_conn(db_path)
    conn.execute(
        "INSERT OR REPLACE INTO runs (run_id, run_name, config_json, created_at) VALUES (?, ?, ?, ?)",
        (run_id, run_name, json.dumps(config), time.time()),
    )
    conn.commit()
    conn.close()


def log_metrics(db_path: str, run_id: str, epoch: int, train_loss: float,
                 train_acc: float, val_loss: float, val_acc: float):
    conn = get_conn(db_path)
    conn.execute(
        "INSERT INTO metrics (run_id, epoch, train_loss, train_acc, val_loss, val_acc) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (run_id, epoch, train_loss, train_acc, val_loss, val_acc),
    )
    conn.commit()
    conn.close()


def get_latest_run_id(db_path: str) -> str:
    conn = get_conn(db_path)
    row = conn.execute("SELECT run_id FROM runs ORDER BY created_at DESC LIMIT 1").fetchone()
    conn.close()
    if row is None:
        raise ValueError("No runs found in database yet. Run src/train.py first.")
    return row[0]


def get_run_config(db_path: str, run_id: str) -> dict:
    conn = get_conn(db_path)
    row = conn.execute("SELECT config_json FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    conn.close()
    if row is None:
        raise ValueError(f"Run {run_id} not found")
    return json.loads(row[0])


def get_metrics(db_path: str, run_id: str):
    """Returns lists: epochs, train_loss, train_acc, val_loss, val_acc"""
    conn = get_conn(db_path)
    rows = conn.execute(
        "SELECT epoch, train_loss, train_acc, val_loss, val_acc FROM metrics "
        "WHERE run_id = ? ORDER BY epoch ASC",
        (run_id,),
    ).fetchall()
    conn.close()
    epochs = [r[0] for r in rows]
    train_loss = [r[1] for r in rows]
    train_acc = [r[2] for r in rows]
    val_loss = [r[3] for r in rows]
    val_acc = [r[4] for r in rows]
    return epochs, train_loss, train_acc, val_loss, val_acc


def list_all_runs(db_path: str):
    conn = get_conn(db_path)
    rows = conn.execute("SELECT run_id, run_name, created_at FROM runs ORDER BY created_at ASC").fetchall()
    conn.close()
    return rows
