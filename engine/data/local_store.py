import json
import os
from pathlib import Path
from typing import Optional


class LocalStore:
    def __init__(self, store_root: str = "engine/data/store"):
        self.root = Path(store_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.video_root = self.root / "videos"
        self.global_root = self.root / "global"
        self.video_root.mkdir(exist_ok=True)
        self.global_root.mkdir(exist_ok=True)

    def video_dir(self, video_id: str) -> Path:
        d = self.video_root / video_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_json(self, video_id: str, filename: str, data) -> None:
        path = self.video_dir(video_id) / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def load_json(self, video_id: str, filename: str) -> Optional[dict]:
        path = self.video_dir(video_id) / filename
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def append_jsonl(self, video_id: str, filename: str, entry: dict) -> None:
        path = self.video_dir(video_id) / filename
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def save_video_analysis(self, video_id: str, analysis: dict) -> None:
        for filename in ["metadata.json", "transcript.json", "segments.json",
                          "labels.json", "candidates.json", "patterns.json",
                          "confidence.json"]:
            key = filename.replace(".json", "")
            data = analysis.get(key)
            if data is not None:
                self.save_json(video_id, filename, data)

        decisions = analysis.get("decisions", [])
        for entry in decisions:
            self.append_jsonl(video_id, "decisions.jsonl", entry)

    def update_global(self, data_type: str, data: dict) -> None:
        filepath = self.global_root / f"{data_type}.json"
        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                existing = json.load(f)
            existing.update(data)
        else:
            existing = data
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, default=str)

    def load_global_pattern_db(self) -> dict:
        filepath = self.global_root / "pattern_db.json"
        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"patterns": [], "versions": [], "meta_edges": []}

    def list_videos(self) -> list[str]:
        if not self.video_root.exists():
            return []
        return sorted([d.name for d in self.video_root.iterdir() if d.is_dir()])
