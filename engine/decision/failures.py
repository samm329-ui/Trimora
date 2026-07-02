import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

from ..scoring.scorer import ScoredClip
from ..data.models import Segment


@dataclass
class FailedCandidate:
    candidate_id: str
    video_id: str
    hook_segment: Optional[dict]
    body_segments: list[dict]
    ending_segment: Optional[dict]
    total_duration: float
    hook_score: float
    body_score: float
    ending_score: float
    flow_score: float
    total_score: float
    rejection_reason: str
    rejection_stage: str
    rules_failed: list[str]
    rules_passed: list[str]
    llm_label: Optional[dict]
    decision_chain: list[dict]
    created_at: str = ""
    pipeline_version: str = "1.0.0"

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str, indent=2)


class FailureStore:
    def __init__(self, video_id: str = "", storage_path: Optional[str] = None):
        self.video_id = video_id
        self.storage_path = storage_path
        self.failures: list[FailedCandidate] = []
        self._total_decisions: int = 0

    def record_decision(self, accepted: bool = True) -> None:
        self._total_decisions += 1

    def record(
        self,
        scored_clip: ScoredClip,
        rules_failed: list[str],
        rules_passed: list[str],
        rejection_reason: str = "rule_engine",
        rejection_stage: str = "scoring",
        llm_label: Optional[dict] = None,
        decision_chain: Optional[list[dict]] = None,
    ) -> FailedCandidate:
        clip = scored_clip.clip
        failure = FailedCandidate(
            candidate_id=f"{clip.hook_segment.id}_{clip.ending_segment.id}_rejected",
            video_id=self.video_id,
            hook_segment=clip.hook_segment.to_dict() if hasattr(clip.hook_segment, 'to_dict') else {"text": clip.hook_segment.text},
            body_segments=[s.to_dict() if hasattr(s, 'to_dict') else {"text": s.text} for s in clip.body_segments],
            ending_segment=clip.ending_segment.to_dict() if hasattr(clip.ending_segment, 'to_dict') else {"text": clip.ending_segment.text},
            total_duration=clip.total_duration,
            hook_score=scored_clip.hook_score,
            body_score=scored_clip.body_score,
            ending_score=scored_clip.ending_score,
            flow_score=scored_clip.flow_score,
            total_score=scored_clip.total_score,
            rejection_reason=rejection_reason,
            rejection_stage=rejection_stage,
            rules_failed=rules_failed,
            rules_passed=rules_passed,
            llm_label=llm_label,
            decision_chain=decision_chain or [],
        )
        self.failures.append(failure)
        return failure

    def get_failure_rate(self) -> float:
        if self._total_decisions == 0:
            return 0.0
        return len(self.failures) / self._total_decisions

    def record_decision(self, accepted: bool = True) -> None:
        self._total_decisions += 1

    def get_top_rejection_reasons(self, n: int = 5) -> list[tuple[str, int]]:
        counts: dict[str, int] = {}
        for f in self.failures:
            counts[f.rejection_reason] = counts.get(f.rejection_reason, 0) + 1
        return sorted(counts.items(), key=lambda x: x[1], reverse=True)[:n]

    def get_rules_failure_rate(self) -> dict[str, float]:
        counts: dict[str, int] = {}
        for f in self.failures:
            for rule in f.rules_failed:
                counts[rule] = counts.get(rule, 0) + 1
        total = len(self.failures)
        return {r: c / total for r, c in counts.items()} if total > 0 else {}

    def summary(self) -> dict:
        return {
            "video_id": self.video_id,
            "total_failures": len(self.failures),
            "top_rejection_reasons": self.get_top_rejection_reasons(),
            "rules_failure_rate": self.get_rules_failure_rate(),
        }

    def record_hard_rejection(
        self,
        segment: Segment,
        rule_name: str,
        rule_category: str,
        reason: str,
        signals: Optional[dict] = None,
    ) -> None:
        from .log import DecisionEntry
        entry = DecisionEntry(
            entity_id=segment.id,
            entity_type="segment",
            stage="rules",
            rule_name=rule_name,
            rule_category=rule_category,
            confidence=0.0,
            outcome="rejected",
            rejection_reason=reason,
            contributing_signals=signals or {},
            video_id=self.video_id,
        )
        self.failures.append(FailedCandidate(
            candidate_id=f"{segment.id}_hard_rejected",
            video_id=self.video_id,
            hook_segment={"text": segment.text, "id": segment.id},
            body_segments=[],
            ending_segment=None,
            total_duration=segment.duration,
            hook_score=0.0,
            body_score=0.0,
            ending_score=0.0,
            flow_score=0.0,
            total_score=0.0,
            rejection_reason=reason,
            rejection_stage="rules",
            rules_failed=[rule_name],
            rules_passed=[],
            llm_label=None,
            decision_chain=[entry.to_dict()],
        ))

    def save(self, filepath: Optional[str] = None) -> None:
        path = filepath or self.storage_path
        if path:
            import os
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    {"video_id": self.video_id, "failures": [fc.to_dict() for fc in self.failures]},
                    f, indent=2, default=str
                )
