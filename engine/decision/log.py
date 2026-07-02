import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional


@dataclass
class DecisionEntry:
    entity_id: str
    entity_type: str
    stage: str
    rule_name: str
    rule_category: str
    confidence: float
    outcome: str
    rejection_reason: str = ""
    contributing_signals: dict = field(default_factory=dict)
    video_id: str = ""
    pipeline_version: str = "1.0.0"
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LogEntry:
    event: str
    video_id: str = ""
    pipeline_version: str = "1.0.0"
    stage: str = ""
    description: str = ""
    data: dict = field(default_factory=dict)
    level: str = "info"
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_json(self) -> str:
        payload = {
            "timestamp": self.timestamp,
            "event": self.event,
            "level": self.level,
            "video_id": self.video_id,
            "pipeline_version": self.pipeline_version,
            "stage": self.stage,
        }
        if self.description:
            payload["description"] = self.description
        payload.update(self.data)
        return json.dumps(payload, default=str)


class DecisionLog:
    def __init__(self, video_id: str = "", pipeline_version: str = "1.0.0",
                 log_dir: Optional[str] = None):
        self.video_id = video_id
        self.pipeline_version = pipeline_version
        self.entries: list[DecisionEntry] = []
        self.log_dir = log_dir

    def record(self, entity_id: str, entity_type: str, stage: str,
               rule_name: str, rule_category: str, confidence: float,
               outcome: str, rejection_reason: str = "",
               contributing_signals: Optional[dict] = None) -> DecisionEntry:
        entry = DecisionEntry(
            entity_id=entity_id,
            entity_type=entity_type,
            stage=stage,
            rule_name=rule_name,
            rule_category=rule_category,
            confidence=confidence,
            outcome=outcome,
            rejection_reason=rejection_reason,
            contributing_signals=contributing_signals or {},
            video_id=self.video_id,
            pipeline_version=self.pipeline_version,
        )
        self.entries.append(entry)
        return entry

    def log_event(self, event: str, level: str = "info",
                  stage: str = "", description: str = "",
                  **data) -> LogEntry:
        entry = LogEntry(
            event=event,
            video_id=self.video_id,
            pipeline_version=self.pipeline_version,
            stage=stage,
            description=description,
            data=data,
            level=level,
        )
        return entry

    def get_by_stage(self, stage: str) -> list[DecisionEntry]:
        return [e for e in self.entries if e.stage == stage]

    def get_by_outcome(self, outcome: str) -> list[DecisionEntry]:
        return [e for e in self.entries if e.outcome == outcome]

    def get_by_entity(self, entity_id: str) -> list[DecisionEntry]:
        return [e for e in self.entries if e.entity_id == entity_id]

    def summary(self) -> dict:
        stages = {}
        for e in self.entries:
            stages.setdefault(e.stage, {"selected": 0, "rejected": 0, "candidate": 0})
            stages[e.stage][e.outcome] = stages[e.stage].get(e.outcome, 0) + 1
        return {
            "video_id": self.video_id,
            "total_decisions": len(self.entries),
            "by_stage": stages,
        }

    def to_json(self) -> str:
        return json.dumps({
            "video_id": self.video_id,
            "pipeline_version": self.pipeline_version,
            "entries": [e.to_dict() for e in self.entries],
        }, default=str, indent=2)

    def save(self, filepath: Optional[str] = None) -> None:
        path = filepath or self.log_dir
        if path and os.path.isdir(path):
            path = os.path.join(path, f"decision_log_{self.video_id or 'unknown'}.json")
        if path:
            os.makedirs(os.path.dirname(path) if path else ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.to_json())
