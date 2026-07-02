from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional


@dataclass
class FeatureProvenance:
    feature_name: str
    value: float
    source: str
    source_confidence: float
    pipeline_version: str = "1.0.0"
    model_version: str = ""
    override_count: int = 0
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return asdict(self)


PROVENANCE_REGISTRY: dict[str, dict] = {
    "hook_strength": {"source": "rule_engine", "source_confidence": 0.92, "replaceable_by": "pattern_match"},
    "shareability": {"source": "llm_teacher", "source_confidence": 0.88, "replaceable_by": "trained_regressor"},
    "speech_rate": {"source": "librosa", "source_confidence": 0.99, "replaceable_by": ""},
    "emotion": {"source": "vader", "source_confidence": 0.76, "replaceable_by": "fine_tuned_classifier"},
    "sentiment": {"source": "vader", "source_confidence": 0.82, "replaceable_by": "fine_tuned_classifier"},
    "requires_context": {"source": "llm_teacher", "source_confidence": 0.91, "replaceable_by": "pattern_match"},
    "llm_confidence": {"source": "llm_teacher", "source_confidence": 0.93, "replaceable_by": "calibrated_model"},
}

DEFAULT_STAGE_RELIABILITIES: dict[str, float] = {
    "transcription": 0.98,
    "segmentation": 0.97,
    "feature_extraction": 0.97,
    "knowledge_graph": 0.96,
    "rule_engine": 0.95,
    "scoring": 0.94,
    "llm_teacher": 0.93,
}


class ConfidencePropagator:
    def __init__(self, stage_reliabilities: Optional[dict[str, float]] = None):
        self.stage_reliabilities = stage_reliabilities or DEFAULT_STAGE_RELIABILITIES

    def propagate(self, stage_confidences: dict[str, float]) -> float:
        cumulative = 1.0
        for stage, stage_confidence in stage_confidences.items():
            reliability = self.stage_reliabilities.get(stage, 0.95)
            cumulative *= (stage_confidence * reliability)
        return cumulative

    def get_downstream_impact(self, upstream_drop: float,
                               stages_remaining: int) -> float:
        avg_reliability = 0.96
        return upstream_drop * (avg_reliability ** stages_remaining)

    def get_feature_provenance(self, feature_name: str) -> Optional[dict]:
        return PROVENANCE_REGISTRY.get(feature_name)
