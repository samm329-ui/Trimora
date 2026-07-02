import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AudioConfig:
    SAMPLE_RATE: int = 16000
    CHANNELS: int = 1
    FORMAT: str = "wav"
    FFMPEG_AUDIO_CODEC: str = "pcm_s16le"
    NORMALIZE: bool = False


@dataclass
class QualityConfig:
    SNR_STRICT_THRESHOLD: float = 10.0
    SPEECH_RATE_HIGH_THRESHOLD: float = 3.5
    DEFAULT_SNR_DB: float = 15.0
    DEFAULT_SPEECH_RATE: float = 2.0
    DEFAULT_VOLUME_RMS: float = 0.05


@dataclass
class ChunkingConfig:
    CHUNK_SIZE_DEFAULT: int = 30
    CHUNK_OVERLAP_DEFAULT: int = 2
    CHUNK_SIZE_STRICT: int = 15
    CHUNK_OVERLAP_STRICT: int = 3
    FADE_IN_MS: int = 50
    FADE_OUT_MS: int = 50
    EXPORT_FORMAT: str = "mp3"


@dataclass
class TranscriptionConfig:
    GROQ_MODEL: str = "whisper-large-v3"
    GROQ_API_KEY_ENV: str = "GROQ_API_KEY"
    RESPONSE_FORMAT: str = "text"
    TEMPERATURE: float = 0.0
    RETRY_COUNT: int = 1


@dataclass
class AlignmentConfig:
    WHISPERX_DEVICE: str = "cpu"
    WHISPERX_BATCH_SIZE: int = 4
    WHISPERX_COMPUTE_TYPE: str = "int8"
    ALIGN_MODEL_LANGUAGE_MAP: dict = field(default_factory=lambda: {
        "english": "en",
        "hindi": "hi",
        "hinglish": "hi",
        "bengali": "bn",
        "spanish": "es",
    })
    UNALIGNED_FALLBACK_SCORE: float = 0.5


@dataclass
class SegmentationConfig:
    MAX_SEGMENT_DURATION: float = 8.0
    MIN_SEGMENT_DURATION: float = 2.0
    MERGE_THRESHOLD: float = 2.0
    PUNCTUATION_BOUNDARIES: tuple = (".", "!", "?", "\n")
    SPEAKER_CHANGE_GAP: float = 0.5


@dataclass
class ScoringConfig:
    HOOK_MIN_SCORE: float = 50.0
    ENDING_MIN_SCORE: float = 40.0
    CLIP_MIN_DURATION: float = 45.0
    CLIP_MAX_DURATION: float = 90.0
    HOOK_MIN_DURATION: float = 3.0
    HOOK_MAX_DURATION: float = 8.0
    BODY_MIN_DURATION: float = 15.0
    BODY_MAX_DURATION: float = 30.0
    ENDING_MIN_DURATION: float = 5.0
    ENDING_MAX_DURATION: float = 10.0
    ENDING_LOOSE_MIN: float = 3.0
    ENDING_LOOSE_MAX: float = 15.0
    HOOK_SPEECH_RATE_MIN: float = 2.0
    HOOK_VOLUME_DELTA_MIN: float = 1.5
    HOOK_ENERGY_SPEECH_RATE: float = 2.5
    HOOK_ENERGY_VOLUME_DELTA: float = 2.0
    TEMPORAL_WINDOW_SECONDS: float = 60.0
    MATCH_WINDOW: int = 5
    RECENCY_THRESHOLD: float = 0.7
    SENTIMENT_CONTRAST_THRESHOLD: float = 0.3
    STITCHING_DIVERSITY_MIN: float = 30.0
    STITCHING_DIVERSITY_MAX: float = 180.0
    STITCHING_DIVERSITY_GOOD: float = 60.0
    WEIGHT_HOOK: float = 0.35
    WEIGHT_BODY: float = 0.25
    WEIGHT_ENDING: float = 0.20
    WEIGHT_FLOW: float = 0.15
    WEIGHT_PRACTICALITY: float = 0.05
    WEIGHT_UNIQUENESS: float = 0.05


@dataclass
class RuleScoreConfig:
    HOOK_ENERGY_SPEECH: float = 20.0
    HOOK_ENERGY_VOLUME: float = 15.0
    HOOK_CURIOSITY: float = 25.0
    HOOK_PROBLEM: float = 20.0
    HOOK_CONTRAST: float = 15.0
    HOOK_ENERGY_BONUS: float = 25.0
    BODY_TEMPORAL_BONUS: float = 20.0
    BODY_BASELINE: float = 50.0
    ENDING_POSITIVE: float = 25.0
    ENDING_TAKEAWAY: float = 30.0
    ENDING_SUMMARY: float = 20.0
    ENDING_DURATION_FIT: float = 15.0
    ENDING_DURATION_LOOSE: float = 10.0
    ENDING_RECENCY: float = 20.0
    ENDING_PRACTICALITY: float = 25.0
    ENDING_RELATABLE: float = 10.0
    ENDING_RESOLUTION_BONUS: float = 30.0
    SOFT_EMOTIONAL_ARC: float = 20.0
    SOFT_PRACTICAL_VALUE: float = 15.0
    SOFT_RELATABLE: float = 10.0


@dataclass
class GraphConfig:
    FOLLOWS_WEIGHT: float = 1.0
    EXPLAINS_WEIGHT_MIN: float = 0.7
    EXPLAINS_WEIGHT_MAX: float = 0.9
    CONTRASTS_WEIGHT_MIN: float = 0.7
    CONTRASTS_WEIGHT_MAX: float = 0.9
    CONCLUDES_WEIGHT_MIN: float = 0.6
    CONCLUDES_WEIGHT_MAX: float = 0.8
    SUPPORTS_WEIGHT_MIN: float = 0.5
    SUPPORTS_WEIGHT_MAX: float = 0.7
    TEMPORAL_WINDOW_EDGES: int = 5
    BFS_MAX_DURATION: float = 90.0


@dataclass
class ConfidenceConfig:
    USE_LOCAL_THRESHOLD: float = 0.9
    USE_PATTERN_THRESHOLD: float = 0.6
    STAGE_RELIABILITY_TRANSCRIPTION: float = 0.98
    STAGE_RELIABILITY_SEGMENTATION: float = 0.97
    STAGE_RELIABILITY_FEATURES: float = 0.97
    STAGE_RELIABILITY_GRAPH: float = 0.96
    STAGE_RELIABILITY_RULES: float = 0.95
    STAGE_RELIABILITY_SCORING: float = 0.94
    STAGE_RELIABILITY_LLM: float = 0.93


@dataclass
class PatternConfig:
    MIN_PATTERN_LEN: int = 3
    BASE_DECAY_RATE: float = 0.0003
    FRESHNESS_HALF_LIFE: int = 365
    NEW_CONFIDENCE_INIT: float = 0.50
    MAX_CONFIDENCE: float = 0.99
    MIN_CONFIDENCE: float = 0.01
    BOOST_ON_SUCCESS: float = 0.02
    PERFORMANCE_THRESHOLD: float = 1.1
    CATEGORY_SIMILARITY_THRESHOLD: float = 0.6


@dataclass
class StorageConfig:
    STORE_ROOT: str = os.path.join(os.path.dirname(__file__), "data", "store")
    STATE_DIR: str = "state"
    TEMP_DIR: str = "temp"
    VIDEO_ROOT: str = "videos"
    GLOBAL_ROOT: str = "global"
    INDEX_FILE: str = "index.json"
    PIPELINE_VERSION: str = "1.0.0"


@dataclass
class LLMConfig:
    USE_LLM: bool = True
    MAX_SEGMENTS_TO_LABEL: int = 100
    MAX_CANDIDATES_TO_LABEL: int = 20
    MAX_REJECTED_TO_LABEL: int = 20
    PROVIDER: str = "groq"
    GROQ_API_KEY_ENV: str = "GROQ_API_KEY"
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    GROQ_TEMPERATURE: float = 0.1
    GROQ_MAX_TOKENS: int = 4096


@dataclass
class PipelineConfig:
    MAX_CONCURRENT_JOBS: int = 5
    AUTO_CLEANUP_HOURS: int = 1
    PARALLEL_CHUNK_TRANSCRIPTION: bool = True
    PARALLEL_MAX_WORKERS: int = 4
    RESUME_ENABLED: bool = True


@dataclass
class EngineConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    rule_scores: RuleScoreConfig = field(default_factory=RuleScoreConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    pattern: PatternConfig = field(default_factory=PatternConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)


_default_config = None


def get_config() -> EngineConfig:
    global _default_config
    if _default_config is None:
        _default_config = EngineConfig()
    return _default_config


def load_config_from_yaml(path: Optional[str] = None) -> EngineConfig:
    cfg = get_config()
    if path and os.path.exists(path):
        import yaml
        with open(path, "r") as f:
            overrides = yaml.safe_load(f)
        if overrides:
            _apply_overrides(cfg, overrides)
    return cfg


def _apply_overrides(cfg: EngineConfig, overrides: dict):
    for section, values in overrides.items():
        section_obj = getattr(cfg, section, None)
        if section_obj:
            for key, val in values.items():
                if hasattr(section_obj, key.upper()):
                    setattr(section_obj, key.upper(), val)
