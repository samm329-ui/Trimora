import json
import sqlite3
import os
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone


class Storage:
    def __init__(self, db_path: str = "engine/data/store/trimora.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _init_db(self) -> None:
        conn = self._connect()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS videos (
                video_id TEXT PRIMARY KEY,
                title TEXT,
                duration REAL,
                audio_quality TEXT,
                language TEXT,
                processed_at TIMESTAMP,
                pipeline_version TEXT
            );

            CREATE TABLE IF NOT EXISTS segments (
                id TEXT PRIMARY KEY,
                video_id TEXT,
                "index" INTEGER,
                text TEXT,
                start REAL,
                end REAL,
                duration REAL,
                speaker TEXT,
                sentiment REAL,
                speech_rate REAL,
                volume_delta REAL,
                patterns TEXT,
                rules_matched TEXT,
                created_at TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS candidates (
                id TEXT PRIMARY KEY,
                video_id TEXT,
                hook_segment_id TEXT,
                body_segment_ids TEXT,
                ending_segment_id TEXT,
                total_duration REAL,
                hook_score REAL,
                body_score REAL,
                ending_score REAL,
                flow_score REAL,
                total_score REAL,
                diversity_score REAL,
                accepted INTEGER,
                rejection_reason TEXT,
                created_at TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT,
                entity_id TEXT,
                entity_type TEXT,
                stage TEXT,
                rule_name TEXT,
                rule_category TEXT,
                confidence REAL,
                outcome TEXT,
                rejection_reason TEXT,
                timestamp TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS labels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT,
                entity_id TEXT,
                entity_type TEXT,
                label_type TEXT,
                label_data TEXT,
                confidence REAL,
                created_at TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS global_edges (
                source_type TEXT,
                target_type TEXT,
                occurrences INTEGER,
                avg_watch_time REAL,
                avg_saves REAL,
                avg_shares REAL,
                llm_confidence REAL,
                avg_confidence REAL,
                first_seen TIMESTAMP,
                last_seen TIMESTAMP,
                PRIMARY KEY (source_type, target_type)
            );

            CREATE TABLE IF NOT EXISTS words (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                segment_id TEXT,
                word_index INTEGER,
                text TEXT,
                start_time REAL,
                end_time REAL,
                confidence REAL,
                is_power BOOLEAN,
                power_category TEXT
            );

            CREATE TABLE IF NOT EXISTS relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT,
                target_id TEXT,
                edge_type TEXT,
                weight REAL,
                evidence TEXT
            );

            CREATE TABLE IF NOT EXISTS failed_candidates (
                id TEXT PRIMARY KEY,
                video_id TEXT,
                hook_segment_id TEXT,
                body_segment_ids TEXT,
                ending_segment_id TEXT,
                total_duration REAL,
                hook_score REAL,
                body_score REAL,
                ending_score REAL,
                flow_score REAL,
                total_score REAL,
                rejection_reason TEXT,
                rejection_stage TEXT,
                rules_failed TEXT,
                rules_passed TEXT,
                llm_label TEXT,
                decision_chain TEXT,
                created_at TIMESTAMP,
                pipeline_version TEXT
            );

            CREATE TABLE IF NOT EXISTS feature_provenance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT,
                entity_id TEXT,
                entity_type TEXT,
                feature_name TEXT,
                value REAL,
                source TEXT,
                source_confidence REAL,
                pipeline_version TEXT,
                created_at TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS pattern_nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_id INTEGER,
                node_index INTEGER,
                node_type TEXT,
                emotion TEXT,
                avg_duration REAL,
                avg_sentiment REAL,
                avg_position REAL,
                occurrences INTEGER,
                confidence REAL
            );

            CREATE TABLE IF NOT EXISTS pattern_edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_id INTEGER,
                source_index INTEGER,
                target_index INTEGER,
                transition_type TEXT,
                transition_quality REAL,
                transition_probability REAL
            );

            CREATE TABLE IF NOT EXISTS meta_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_pattern_id INTEGER,
                target_pattern_id INTEGER,
                category TEXT,
                occurrences INTEGER,
                transition_probability REAL,
                avg_saves REAL,
                avg_shares REAL,
                confidence REAL
            );
        """)
        conn.commit()

    def save_video(self, video_id: str, title: str = "",
                   duration: float = 0.0, audio_quality: str = "",
                   language: str = "", pipeline_version: str = "") -> None:
        conn = self._connect()
        conn.execute("""
            INSERT OR REPLACE INTO videos
            (video_id, title, duration, audio_quality, language, processed_at, pipeline_version)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (video_id, title, duration, audio_quality, language,
              datetime.now(timezone.utc).isoformat(), pipeline_version))
        conn.commit()

    def save_segment(self, video_id: str, segment: dict) -> None:
        conn = self._connect()
        conn.execute("""
            INSERT OR REPLACE INTO segments
            (id, video_id, "index", text, start, end, duration, speaker,
             sentiment, speech_rate, volume_delta, patterns, rules_matched, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            segment.get("id"), video_id, segment.get("index", 0),
            segment.get("text", ""), segment.get("start", 0),
            segment.get("end", 0), segment.get("duration", 0),
            segment.get("speaker", ""), segment.get("sentiment", 0),
            segment.get("speech_rate", 0), segment.get("volume_delta", 0),
            json.dumps(segment.get("patterns", [])),
            json.dumps(segment.get("rules_matched", [])),
            datetime.now(timezone.utc).isoformat(),
        ))
        conn.commit()

    def save_candidate(self, video_id: str, candidate: dict) -> None:
        conn = self._connect()
        conn.execute("""
            INSERT OR REPLACE INTO candidates
            (id, video_id, hook_segment_id, body_segment_ids, ending_segment_id,
             total_duration, hook_score, body_score, ending_score, flow_score,
             total_score, diversity_score, accepted, rejection_reason, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            candidate.get("id"), video_id,
            candidate.get("hook_segment_id"),
            json.dumps(candidate.get("body_segment_ids", [])),
            candidate.get("ending_segment_id"),
            candidate.get("total_duration", 0),
            candidate.get("hook_score", 0),
            candidate.get("body_score", 0),
            candidate.get("ending_score", 0),
            candidate.get("flow_score", 0),
            candidate.get("total_score", 0),
            candidate.get("diversity_score", 0),
            1 if candidate.get("accepted") else 0,
            candidate.get("rejection_reason", ""),
            datetime.now(timezone.utc).isoformat(),
        ))
        conn.commit()

    def save_decision(self, video_id: str, entity_id: str,
                      entity_type: str, stage: str, rule_name: str,
                      rule_category: str, confidence: float,
                      outcome: str, rejection_reason: str = "") -> None:
        conn = self._connect()
        conn.execute("""
            INSERT INTO decisions
            (video_id, entity_id, entity_type, stage, rule_name, rule_category,
             confidence, outcome, rejection_reason, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (video_id, entity_id, entity_type, stage, rule_name, rule_category,
              confidence, outcome, rejection_reason,
              datetime.now(timezone.utc).isoformat()))
        conn.commit()

    def get_video(self, video_id: str) -> Optional[dict]:
        conn = self._connect()
        row = conn.execute("SELECT * FROM videos WHERE video_id = ?", (video_id,)).fetchone()
        return dict(row) if row else None

    def get_segments(self, video_id: str) -> list[dict]:
        conn = self._connect()
        rows = conn.execute(
            'SELECT * FROM segments WHERE video_id = ? ORDER BY "index"', (video_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_candidates(self, video_id: str, accepted_only: bool = False) -> list[dict]:
        conn = self._connect()
        query = "SELECT * FROM candidates WHERE video_id = ?"
        params = [video_id]
        if accepted_only:
            query += " AND accepted = 1"
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def save_failed_candidate(self, video_id: str, fc: dict) -> None:
        conn = self._connect()
        conn.execute("""
            INSERT OR REPLACE INTO failed_candidates
            (id, video_id, hook_segment_id, body_segment_ids, ending_segment_id,
             total_duration, hook_score, body_score, ending_score, flow_score,
             total_score, rejection_reason, rejection_stage, rules_failed,
             rules_passed, llm_label, decision_chain, created_at, pipeline_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            fc.get("candidate_id", ""), video_id,
            fc.get("hook_segment_id", ""),
            json.dumps(fc.get("body_segment_ids", [])),
            fc.get("ending_segment_id", ""),
            fc.get("total_duration", 0), fc.get("hook_score", 0),
            fc.get("body_score", 0), fc.get("ending_score", 0),
            fc.get("flow_score", 0), fc.get("total_score", 0),
            fc.get("rejection_reason", ""), fc.get("rejection_stage", ""),
            json.dumps(fc.get("rules_failed", [])),
            json.dumps(fc.get("rules_passed", [])),
            json.dumps(fc.get("llm_label", {}) or {}),
            json.dumps(fc.get("decision_chain", [])),
            datetime.now(timezone.utc).isoformat(),
            fc.get("pipeline_version", ""),
        ))
        conn.commit()

    def save_feature_provenance(self, video_id: str, entity_id: str,
                                 entity_type: str, feature_name: str, value: float,
                                 source: str, source_confidence: float,
                                 pipeline_version: str = "") -> None:
        conn = self._connect()
        conn.execute("""
            INSERT INTO feature_provenance
            (video_id, entity_id, entity_type, feature_name, value,
             source, source_confidence, pipeline_version, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (video_id, entity_id, entity_type, feature_name, value,
              source, source_confidence, pipeline_version,
              datetime.now(timezone.utc).isoformat()))
        conn.commit()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
