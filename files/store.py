"""SQLite audit log of decisions and outcomes (stdlib only)."""
import json
import sqlite3
import time


class Store:
    def __init__(self, path: str = ":memory:"):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("""CREATE TABLE IF NOT EXISTS decisions(
            id INTEGER PRIMARY KEY, ts REAL, slice TEXT, query TEXT, action TEXT, answer TEXT,
            raw_conf REAL, conf REAL, agreement REAL, threshold REAL, reason TEXT, scores TEXT,
            correct INTEGER)""")
        self.db.execute("""CREATE TABLE IF NOT EXISTS unresolved_gaps(
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, decision_id INTEGER, slice TEXT, query TEXT, reason_code TEXT)""")
        self.db.commit()

    def log_gap(self, d, query: str) -> None:
        self.db.execute("INSERT INTO unresolved_gaps(ts,decision_id,slice,query,reason_code) VALUES(?,?,?,?,?)",
                        (time.time(), d.id, d.slice, query, d.reason_code))
        self.db.commit()

    def gaps(self):
        """Abstentions grouped by slice and reason: where the system says it hasn't learned enough."""
        return self.db.execute("SELECT slice, reason_code, COUNT(*) FROM unresolved_gaps "
                               "GROUP BY slice, reason_code ORDER BY COUNT(*) DESC").fetchall()

    def log(self, d, query: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO decisions VALUES(?,?,?,?,?,?,?,?,?,?,?,?,NULL)",
                        (d.id, time.time(), d.slice, query, d.action, d.text, d.raw_confidence,
                         d.confidence, d.agreement, d.threshold, d.reason, json.dumps(d.principle_scores)))
        self.db.commit()

    def label(self, decision_id: int, correct: bool) -> None:
        self.db.execute("UPDATE decisions SET correct=? WHERE id=?", (int(correct), decision_id))
        self.db.commit()
