import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional


class ClipTracker:
    def __init__(self, video_id: str = "", storage_dir: Optional[str] = None):
        self.video_id = video_id
        self.storage_dir = storage_dir
        self.candidates_seen: dict[str, dict] = {}
        self.segment_usage: dict[str, int] = defaultdict(int)
        self.history: list[dict] = []

    def record_candidate(self, candidate_id: str, score: float,
                         hook_id: str, body_ids: list[str], ending_id: str,
                         accepted: bool = False) -> None:
        self.candidates_seen[candidate_id] = {
            "candidate_id": candidate_id,
            "score": score,
            "hook_id": hook_id,
            "body_ids": body_ids,
            "ending_id": ending_id,
            "accepted": accepted,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def record_segment_usage(self, segment_id: str) -> None:
        self.segment_usage[segment_id] += 1

    def record_history(self, event: str, details: Optional[dict] = None) -> None:
        self.history.append({
            "event": event,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": details or {},
            "video_id": self.video_id,
        })

    def get_top_candidates(self, n: int = 5) -> list[dict]:
        sorted_cands = sorted(
            self.candidates_seen.values(),
            key=lambda x: x["score"],
            reverse=True
        )
        return sorted_cands[:n]

    def get_accepted_count(self) -> int:
        return sum(1 for c in self.candidates_seen.values() if c["accepted"])

    def get_rejected_count(self) -> int:
        return sum(1 for c in self.candidates_seen.values() if not c["accepted"])

    def get_most_used_segments(self, n: int = 10) -> list[tuple[str, int]]:
        return sorted(self.segment_usage.items(), key=lambda x: x[1], reverse=True)[:n]

    def save_state(self, filepath: Optional[str] = None) -> None:
        path = filepath or self.storage_dir
        if path and os.path.isdir(path):
            path = os.path.join(path, f"tracker_{self.video_id or 'unknown'}.json")
        if path:
            os.makedirs(os.path.dirname(path) if path else ".", exist_ok=True)
            state = {
                "video_id": self.video_id,
                "candidates_seen": self.candidates_seen,
                "segment_usage": dict(self.segment_usage),
                "history": self.history,
                "exported_at": datetime.now(timezone.utc).isoformat(),
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, default=str)

    def summary(self) -> dict:
        return {
            "video_id": self.video_id,
            "total_candidates": len(self.candidates_seen),
            "accepted": self.get_accepted_count(),
            "rejected": self.get_rejected_count(),
            "unique_segments_used": len(self.segment_usage),
            "most_used_segments": self.get_most_used_segments(5),
            "top_scores": [c["score"] for c in self.get_top_candidates(3)],
        }
