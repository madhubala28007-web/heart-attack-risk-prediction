"""SQLite database : stores patient details and prediction history."""
import sqlite3
from datetime import datetime
from config import DB_PATH


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    with get_connection() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS patients (
            patient_id  INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT    NOT NULL,
            age         INTEGER CHECK (age BETWEEN 1 AND 120),
            gender      TEXT,
            contact     TEXT,
            created_at  TEXT    NOT NULL
        );
        CREATE TABLE IF NOT EXISTS predictions (
            pred_id     INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id  INTEGER NOT NULL,
            image_path  TEXT    NOT NULL,
            risk_prob   REAL    NOT NULL,
            risk_label  TEXT    NOT NULL,
            model_name  TEXT,
            created_at  TEXT    NOT NULL,
            FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
        );
        """)


def add_patient(name, age, gender, contact=""):
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO patients (name, age, gender, contact, created_at) VALUES (?,?,?,?,?)",
            (name, age, gender, contact, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        return cur.lastrowid


def add_prediction(patient_id, image_path, prob, label, model_name="best_model"):
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO predictions (patient_id, image_path, risk_prob, risk_label, "
            "model_name, created_at) VALUES (?,?,?,?,?,?)",
            (patient_id, image_path, float(prob), label, model_name,
             datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        return cur.lastrowid


def list_predictions(limit=20):
    with get_connection() as conn:
        return conn.execute(
            "SELECT p.pred_id, t.name, t.age, p.risk_prob, p.risk_label, p.created_at "
            "FROM predictions p JOIN patients t ON t.patient_id = p.patient_id "
            "ORDER BY p.pred_id DESC LIMIT ?", (limit,)).fetchall()


if __name__ == "__main__":
    init_db()
    print("Database initialised ->", DB_PATH)
