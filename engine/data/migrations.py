import sqlite3
import json
from pathlib import Path
from typing import Optional

from .local_store import LocalStore


class Migration:
    def __init__(self, version: int, description: str, sql: str):
        self.version = version
        self.description = description
        self.sql = sql


MIGRATIONS = [
    Migration(1, "Initial schema", """
        CREATE TABLE IF NOT EXISTS videos (
            video_id TEXT PRIMARY KEY,
            title TEXT, duration REAL, audio_quality TEXT,
            language TEXT, processed_at TIMESTAMP, pipeline_version TEXT
        );
        CREATE TABLE IF NOT EXISTS segments (
            id TEXT PRIMARY KEY, video_id TEXT, "index" INTEGER,
            text TEXT, start REAL, end REAL, duration REAL,
            speaker TEXT, sentiment REAL, speech_rate REAL,
            volume_delta REAL, patterns TEXT, rules_matched TEXT,
            created_at TIMESTAMP
        );
    """),
    Migration(2, "Add candidates table", """
        CREATE TABLE IF NOT EXISTS candidates (
            id TEXT PRIMARY KEY, video_id TEXT,
            hook_segment_id TEXT, body_segment_ids TEXT,
            ending_segment_id TEXT, total_duration REAL,
            hook_score REAL, body_score REAL, ending_score REAL,
            flow_score REAL, total_score REAL, diversity_score REAL,
            accepted INTEGER, rejection_reason TEXT, created_at TIMESTAMP
        );
    """),
    Migration(3, "Add decisions + labels tables", """
        CREATE TABLE IF NOT EXISTS decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, video_id TEXT,
            entity_id TEXT, entity_type TEXT, stage TEXT,
            rule_name TEXT, rule_category TEXT, confidence REAL,
            outcome TEXT, rejection_reason TEXT, timestamp TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS labels (
            id INTEGER PRIMARY KEY AUTOINCREMENT, video_id TEXT,
            entity_id TEXT, entity_type TEXT, label_type TEXT,
            label_data TEXT, confidence REAL, created_at TIMESTAMP
        );
    """),
    Migration(4, "Add global edges table", """
        CREATE TABLE IF NOT EXISTS global_edges (
            source_type TEXT, target_type TEXT, occurrences INTEGER,
            avg_watch_time REAL, avg_saves REAL, avg_shares REAL,
            llm_confidence REAL, avg_confidence REAL,
            first_seen TIMESTAMP, last_seen TIMESTAMP,
            PRIMARY KEY (source_type, target_type)
        );
    """),
    Migration(5, "Add words, relationships, failed_candidates, feature_provenance, pattern_* tables", """
        CREATE TABLE IF NOT EXISTS words (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            segment_id TEXT, word_index INTEGER,
            text TEXT, start_time REAL, end_time REAL,
            confidence REAL, is_power BOOLEAN, power_category TEXT
        );
        CREATE TABLE IF NOT EXISTS relationships (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id TEXT, target_id TEXT,
            edge_type TEXT, weight REAL, evidence TEXT
        );
        CREATE TABLE IF NOT EXISTS failed_candidates (
            id TEXT PRIMARY KEY, video_id TEXT,
            hook_segment_id TEXT, body_segment_ids TEXT,
            ending_segment_id TEXT, total_duration REAL,
            hook_score REAL, body_score REAL, ending_score REAL,
            flow_score REAL, total_score REAL,
            rejection_reason TEXT, rejection_stage TEXT,
            rules_failed TEXT, rules_passed TEXT,
            llm_label TEXT, decision_chain TEXT,
            created_at TIMESTAMP, pipeline_version TEXT
        );
        CREATE TABLE IF NOT EXISTS feature_provenance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id TEXT, entity_id TEXT, entity_type TEXT,
            feature_name TEXT, value REAL,
            source TEXT, source_confidence REAL,
            pipeline_version TEXT, created_at TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS pattern_nodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pattern_id INTEGER, node_index INTEGER,
            node_type TEXT, emotion TEXT,
            avg_duration REAL, avg_sentiment REAL,
            avg_position REAL, occurrences INTEGER, confidence REAL
        );
        CREATE TABLE IF NOT EXISTS pattern_edges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pattern_id INTEGER, source_index INTEGER,
            target_index INTEGER, transition_type TEXT,
            transition_quality REAL, transition_probability REAL
        );
        CREATE TABLE IF NOT EXISTS meta_patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_pattern_id INTEGER, target_pattern_id INTEGER,
            category TEXT, occurrences INTEGER,
            transition_probability REAL, avg_saves REAL,
            avg_shares REAL, confidence REAL
        );
    """),
]


class MigrationRunner:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
        return self._conn

    def get_current_version(self) -> int:
        conn = self._connect()
        try:
            row = conn.execute("SELECT MAX(version) FROM _migrations").fetchone()
            return row[0] if row and row[0] else 0
        except sqlite3.OperationalError:
            conn.execute("CREATE TABLE IF NOT EXISTS _migrations (version INTEGER PRIMARY KEY, description TEXT, applied_at TIMESTAMP)")
            conn.commit()
            return 0

    def run_pending(self) -> list[str]:
        current = self.get_current_version()
        applied = []
        conn = self._connect()
        for m in MIGRATIONS:
            if m.version > current:
                conn.executescript(m.sql)
                conn.execute(
                    "INSERT INTO _migrations (version, description, applied_at) VALUES (?, ?, datetime('now'))",
                    (m.version, m.description)
                )
                conn.commit()
                applied.append(f"v{m.version}: {m.description}")
        return applied

    def import_from_local_store(self, local_store: LocalStore) -> dict:
        conn = self._connect()
        stats = {"videos": 0, "segments": 0, "candidates": 0}

        for video_id in local_store.list_videos():
            metadata = local_store.load_json(video_id, "metadata.json")
            if metadata:
                conn.execute("""
                    INSERT OR REPLACE INTO videos
                    (video_id, title, duration, processed_at, pipeline_version)
                    VALUES (?, ?, ?, datetime('now'), ?)
                """, (video_id, metadata.get("title", ""), metadata.get("duration", 0),
                      metadata.get("pipeline_version", "")))
                stats["videos"] += 1

            segments = local_store.load_json(video_id, "segments.json")
            if segments:
                for seg in segments if isinstance(segments, list) else [segments]:
                    conn.execute("""
                        INSERT OR REPLACE INTO segments
                        (id, video_id, text, start, end, duration, sentiment,
                         speech_rate, volume_delta, patterns, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                    """, (seg.get("id", ""), video_id, seg.get("text", ""),
                          seg.get("start", 0), seg.get("end", 0), seg.get("duration", 0),
                          seg.get("sentiment", 0), seg.get("speech_rate", 0),
                          seg.get("volume_delta", 0),
                          json.dumps(seg.get("patterns", []))))
                    stats["segments"] += 1

        conn.commit()
        return stats
