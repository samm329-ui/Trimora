from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Word:
    text: str
    start: float
    end: float
    confidence: float = 0.0
    is_power: bool = False
    power_category: str = ""


@dataclass
class Segment:
    id: str
    video_id: str = ""
    index: int = 0
    text: str = ""
    words: list[Word] = field(default_factory=list)
    start: float = 0.0
    end: float = 0.0
    duration: float = 0.0
    speaker: str = ""
    sentiment: float = 0.0
    speech_rate: float = 0.0
    pause_after: float = 0.0
    volume_delta: float = 0.0
    patterns: list[str] = field(default_factory=list)
    rules_matched: list[str] = field(default_factory=list)
    llm_analysis: str = ""
    watch_time_pct: float = 0.0
    shares: int = 0
    saves: int = 0


@dataclass
class AudioQuality:
    snr_db: float = 0.0
    speech_rate: float = 0.0
    volume_rms: float = 0.0


@dataclass
class Chunk:
    path: str
    start_time: float
    end_time: float
    index: int = 0


@dataclass
class ProcessedChunk:
    text: str
    start_time: float
    end_time: float
    index: int
    language: str = "english"


@dataclass
class AudioFeatures:
    speech_rate: float = 0.0
    volume: float = 0.0
    volume_delta: float = 0.0
    pause_after: Optional[float] = None
    onset_count: int = 0


@dataclass
class HookCandidate:
    segment_id: str
    score: float
    matched_rules: list[str] = field(default_factory=list)


@dataclass
class BodyCandidate:
    id: str
    score: float


@dataclass
class EndingCandidate:
    id: str
    score: float


@dataclass
class CompleteClip:
    hook_id: str
    body_ids: list[str]
    ending_id: str
    segments: list[Segment] = field(default_factory=list)
    diversity_score: float = 0.0
    rejection_reason: str = ""

    @property
    def total_duration(self) -> float:
        if not self.segments:
            return 0.0
        return sum(s.duration for s in self.segments)

    @property
    def hook_duration(self) -> float:
        if not self.segments:
            return 0.0
        return self.segments[0].duration if self.segments else 0.0

    @property
    def speaker_changes(self) -> int:
        if not self.segments:
            return 0
        speakers = [s.speaker for s in self.segments if s.speaker]
        return sum(1 for i in range(1, len(speakers)) if speakers[i] != speakers[i-1])

    @property
    def hook_segment(self) -> Optional[Segment]:
        return self.segments[0] if self.segments else None

    @property
    def body_segments(self) -> list[Segment]:
        return self.segments[1:-1] if len(self.segments) > 2 else []

    @property
    def ending_segment(self) -> Optional[Segment]:
        return self.segments[-1] if len(self.segments) > 1 else None

    @property
    def all_segments(self) -> list[Segment]:
        return self.segments


@dataclass
class ValidClip:
    hook_segment: Segment
    body_segments: list[Segment]
    ending_segment: Segment
    all_segments: list[Segment] = field(default_factory=list)
    total_duration: float = 0.0
    hook_duration: float = 0.0
    speaker_changes: int = 0
    diversity_score: float = 0.0

    def __post_init__(self):
        if not self.all_segments:
            self.all_segments = (
                [self.hook_segment] + self.body_segments + [self.ending_segment]
            )
        if self.total_duration == 0.0:
            self.total_duration = sum(s.duration for s in self.all_segments)
        if self.hook_duration == 0.0:
            self.hook_duration = self.hook_segment.duration


@dataclass
class ScoredClip:
    clip: ValidClip
    hook_score: float = 0.0
    body_score: float = 0.0
    ending_score: float = 0.0
    flow_score: float = 0.0
    total_score: float = 0.0


@dataclass
class Relationship:
    source_id: str
    target_id: str
    edge_type: str
    weight: float
    evidence: str


@dataclass
class SegmentLabel:
    is_hook: float = 0.0
    hook_strength: float = 0.0
    is_context: float = 0.0
    is_takeaway: float = 0.0
    emotion: str = "neutral"
    requires_previous_context: float = 0.0
    creates_new_context: float = 0.0
    is_story: float = 0.0
    is_opinion: float = 0.0
    is_fact: float = 0.0
    speaker_confidence: float = 0.0
    saveability: float = 0.0
    shareability: float = 0.0
    confidence: float = 0.0


@dataclass
class CandidateLabel:
    story_complete: float = 0.0
    transition_quality: float = 0.0
    context_missing: float = 0.0
    shareability: float = 0.0
    saveability: float = 0.0
    hook_strength: float = 0.0
    ending_strength: float = 0.0
    emotional_arc_build_up: float = 0.0
    naturalness: float = 0.0
    curiosity_gap: float = 0.0
    confidence: float = 0.0


@dataclass
class VideoAnalysis:
    video_id: str
    filename: str
    duration: float
    language: str
    transcript_text: str
    word_count: int
    segment_count: int
    snr_db: float
    speech_rate: float
    volume_rms: float
    segments: list[Segment]
    relationships: list[Relationship]
    candidates: list
    llm_viral_moments: list[dict]
    llm_reasoning: str
    created_at: datetime
    processing_time: float
    pipeline_version: str


@dataclass
class Candidate:
    id: str
    video_id: str
    hook_segment_id: str
    body_segment_ids: list[str]
    ending_segment_id: str
    total_duration: float
    hook_score: float
    body_score: float
    ending_score: float
    flow_score: float
    total_score: float
    rules_passed: list[str]
    rules_failed: list[str]
    llm_ranking: int
    llm_reasoning: str
    actual_performance: dict


@dataclass
class FailedCandidate:
    candidate_id: str
    video_id: str
    hook_segment_id: str
    body_segment_ids: list[str]
    ending_segment_id: str
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
    llm_label: Optional[CandidateLabel] = None
    created_at: datetime = field(default_factory=datetime.now)
    pipeline_version: str = ""
