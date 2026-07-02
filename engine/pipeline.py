import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import get_config
from .decision.log import DecisionLog
from .decision.tracker import ClipTracker
from .decision.failures import FailureStore
from .confidence.scorer import ConfidencePropagator
from .confidence.threshold import AdaptiveThresholdEngine
from .data.local_store import LocalStore
from .data.storage import Storage


@dataclass
class PipelineResult:
    video_id: str
    candidates: list[dict]
    best_clip: Optional[dict]
    stats: dict
    error: Optional[str] = None


class PipelineError(Exception):
    def __init__(self, stage: str, message: str, recoverable: bool = False):
        self.stage = stage
        self.recoverable = recoverable
        super().__init__(f"[{stage}] {message}")


class Pipeline:
    STATE_FILE = "state.json"

    def __init__(self, video_id: str = ""):
        self.video_id = video_id or str(uuid.uuid4())
        self.cfg = get_config()
        self.decision_log = DecisionLog(video_id=self.video_id)
        self.tracker = ClipTracker(video_id=self.video_id)
        self.failure_store = FailureStore(video_id=self.video_id)
        self.confidence_propagator = ConfidencePropagator()
        self.threshold_engine = AdaptiveThresholdEngine()
        self.local_store = LocalStore()
        self.storage = Storage()
        self._stats: dict = {}
        self._global_graph = None
        self._context_db = None
        self._pattern_engine = None
        self._video_state_dir: Optional[Path] = None

    @property
    def global_graph(self):
        if self._global_graph is None:
            from .knowledge.global_graph import GlobalKnowledgeGraph
            self._global_graph = GlobalKnowledgeGraph()
        return self._global_graph

    @property
    def context_db(self):
        if self._context_db is None:
            from .knowledge.context_db import ContextDatabase
            self._context_db = ContextDatabase()
        return self._context_db

    @property
    def pattern_engine(self):
        if self._pattern_engine is None:
            from .patterns.engine import PatternIntelligenceEngine
            self._pattern_engine = PatternIntelligenceEngine(
                global_graph=self.global_graph,
                context_db=self.context_db,
            )
        return self._pattern_engine

    def _state_dir(self) -> Path:
        if self._video_state_dir is None:
            base = Path(self.cfg.storage.STORE_ROOT) / self.video_id
            base.mkdir(parents=True, exist_ok=True)
            self._video_state_dir = base
        return self._video_state_dir

    def get_completed_stages(self) -> set[str]:
        state_path = self._state_dir() / self.STATE_FILE
        if state_path.exists():
            try:
                with open(state_path, "r") as f:
                    return set(json.load(f).get("completed_stages", []))
            except (json.JSONDecodeError, OSError):
                return set()
        return set()

    def mark_stage_complete(self, stage: str) -> None:
        state_path = self._state_dir() / self.STATE_FILE
        state = {"completed_stages": []}
        if state_path.exists():
            try:
                with open(state_path, "r") as f:
                    state = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        if stage not in state["completed_stages"]:
            state["completed_stages"].append(stage)
        with open(state_path, "w") as f:
            json.dump(state, f)

    def run_stage(self, stage_name: str, fn, *args, **kwargs):
        try:
            result = fn(*args, **kwargs)
            return result
        except PipelineError as e:
            if e.recoverable:
                return self._get_fallback(stage_name)
            raise
        except Exception as e:
            raise PipelineError(stage_name, str(e))

    def _get_fallback(self, stage_name: str):
        fallbacks = {
            "audio_extraction": None,
            "quality_analysis": None,
            "transcription": [],
            "word_alignment": [],
            "feature_extraction": None,
            "llm_teacher": None,
            "pattern_matching": None,
            "global_graph_update": None,
        }
        return fallbacks.get(stage_name, None)

    def run(self, video_path: str, category: str = "") -> PipelineResult:
        start_time = time.time()
        try:
            return self._run_internal(video_path, category, start_time)
        except Exception as e:
            return PipelineResult(
                video_id=self.video_id,
                candidates=[],
                best_clip=None,
                stats={"error": str(e), "processing_time": time.time() - start_time},
                error=str(e),
            )

    def _run_internal(self, video_path: str, category: str,
                       start_time: float) -> PipelineResult:
        # Lazy imports for modules with optional dependencies
        from .audio.extractor import extract_audio
        from .audio.quality import measure_audio_quality
        from .audio.chunker import overlap_chunk
        from .transcription.transcriber import transcribe_chunk
        from .transcription.language import detect_language
        from .transcription.fillers import remove_fillers
        from .transcription.merger import merge_chunks
        from .segmentation.segmenter import split_into_atomic_segments
        from .features.sentiment import compute_sentiment
        from .features.patterns import match_patterns
        from .features.structural import compute_structural_features
        from .graph.knowledge_graph import KnowledgeGraph
        from .graph.relationships import detect_relationships as detect_kg_edges
        from .knowledge.local_graph import LocalKnowledgeGraph
        from .knowledge.relationships import detect_relationships as detect_local_edges
        from .scoring.candidate_generator import generate_candidates
        from .scoring.rule_engine import validate_clip
        from .scoring.scorer import make_valid_clip, score_clip
        from .rules.hook_rules import find_hook_candidates

        # Optional: word alignment (requires whisperx — disabled by default, CPU bottleneck)
        align_segments = None

        # Optional: audio features (requires librosa)
        try:
            from .features.audio_features import compute_audio_features
        except ImportError:
            compute_audio_features = None

        completed = self.get_completed_stages()

        # Stage 1: Extract audio
        if "audio_extraction" not in completed:
            audio_path = self.run_stage("audio_extraction", extract_audio,
                                         video_path, f"{self.video_id}.{self.cfg.audio.FORMAT}")
            self.mark_stage_complete("audio_extraction")
        else:
            audio_path = f"{self.video_id}.{self.cfg.audio.FORMAT}"
        self._stats["audio_extracted"] = True

        # Stage 2: Audio quality
        if "quality_analysis" not in completed:
            quality = self.run_stage("quality_analysis", measure_audio_quality, audio_path)
            self.mark_stage_complete("quality_analysis")
        else:
            from .data.models import AudioQuality
            quality = AudioQuality(snr_db=self.cfg.quality.DEFAULT_SNR_DB,
                                   speech_rate=self.cfg.quality.DEFAULT_SPEECH_RATE,
                                   volume_rms=self.cfg.quality.DEFAULT_VOLUME_RMS)
        self._stats["snr_db"] = round(quality.snr_db, 2)

        # Stage 3: Chunk audio
        if "chunking" not in completed:
            chunks = self.run_stage("chunking", overlap_chunk, audio_path, quality)
            self.mark_stage_complete("chunking")
        else:
            chunks = []
        self._stats["chunks"] = len(chunks)

        # Stage 4: Transcribe chunks (parallel)
        if "transcription" not in completed:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            max_workers = self.cfg.pipeline.PARALLEL_MAX_WORKERS
            raw_by_index = {}
            total = len(chunks)
            done = 0

            print(f"[pipeline] Transcribing {total} chunks ({max_workers} workers)...")
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                fut_map = {pool.submit(transcribe_chunk, chunk.path): chunk for chunk in chunks}
                for fut in as_completed(fut_map):
                    chunk = fut_map[fut]
                    done += 1
                    if done % max(1, total // 10) == 0 or done == total:
                        print(f"[pipeline] Transcribed {done}/{total} chunks")
                    try:
                        text = fut.result()
                    except Exception:
                        continue
                    text = remove_fillers(text)
                    if text.strip():
                        lang = detect_language(text.split())
                        raw_by_index[chunk.index] = {
                            "text": text,
                            "start": chunk.start_time,
                            "end": chunk.end_time,
                            "language": lang,
                        }
            raw_segments = [raw_by_index[i] for i in sorted(raw_by_index)]
            self.mark_stage_complete("transcription")
            print(f"[pipeline] Transcription done: {len(raw_segments)} segments")
        else:
            raw_segments = []
        self._stats["raw_segments"] = len(raw_segments)

        # Stage 5: Word alignment (optional)
        if align_segments is not None:
            aligned = self.run_stage("word_alignment", align_segments,
                                      raw_segments, audio_path, "en")
            self._stats["aligned_segments"] = len(aligned)
        else:
            aligned = raw_segments
            self._stats["aligned_segments"] = 0
            self._stats["word_alignment"] = "unavailable"

        # Build full transcript for scoring uniqueness
        full_transcript = " ".join(s.get("text", "") for s in aligned)

        # Stage 6: Atomic segmentation
        atomic_segments = split_into_atomic_segments(aligned)
        total_duration = max(s.end for s in atomic_segments) if atomic_segments else 0
        self._stats["atomic_segments"] = len(atomic_segments)

        # Stage 7: Feature extraction
        for seg in atomic_segments:
            seg.sentiment = compute_sentiment(seg.text)
            seg.patterns = match_patterns(seg.text)
        self._stats["features_computed"] = len(atomic_segments)

        # Stage 7c: Audio features (if audio available)
        if compute_audio_features is not None and atomic_segments:
            import librosa
            y, sr = librosa.load(audio_path, sr=16000)
            for seg in atomic_segments:
                af = compute_audio_features(seg, y, sr)
                seg.speech_rate = af.speech_rate
                seg.volume_delta = af.volume_delta

        # Stage 7d: Structural features
        structural_features = {
            seg.id: compute_structural_features(seg, total_duration)
            for seg in atomic_segments
        }

        # Stage 8a: Build knowledge graph
        local_kg = LocalKnowledgeGraph()
        detect_local_edges(atomic_segments, local_kg)

        # Stage 8b: Legacy knowledge graph for rules
        kg = KnowledgeGraph()
        detect_kg_edges(atomic_segments, kg)

        # Stage 9-11: Generate candidates via rules
        hooks = find_hook_candidates(kg)
        self._stats["hooks_found"] = len(hooks)

        clips = generate_candidates(hooks, kg)
        self._stats["candidates_generated"] = len(clips)

        # Stage 12: Validate candidates
        accepted_clips = []
        rejected_clips = []
        for clip in clips:
            accepted, passed, failed = validate_clip(clip)
            vc = make_valid_clip(clip)
            scored = score_clip(vc, full_transcript=full_transcript)

            if accepted:
                accepted_clips.append(scored)
            else:
                rejected_clips.append(scored)
                self.failure_store.record(
                    scored, rules_failed=failed, rules_passed=passed
                )
            self.failure_store.record_decision(accepted=accepted)

        self._stats["candidates_passed"] = len(accepted_clips)
        self._stats["candidates_rejected"] = len(rejected_clips)

        # Stage 13: Rank and select best
        accepted_clips.sort(key=lambda c: c.total_score, reverse=True)
        best_scored = accepted_clips[0] if accepted_clips else None

        candidates_data = []
        for sc in accepted_clips[:10]:
            vc = sc.clip
            candidates_data.append({
                "id": f"{vc.hook_segment.id}_{vc.ending_segment.id}",
                "hook_text": vc.hook_segment.text,
                "body_text": " ".join(s.text for s in vc.body_segments),
                "ending_text": vc.ending_segment.text,
                "total_duration": vc.total_duration,
                "hook_score": sc.hook_score,
                "body_score": sc.body_score,
                "ending_score": sc.ending_score,
                "flow_score": sc.flow_score,
                "total_score": sc.total_score,
                "start": vc.hook_segment.start,
                "end": vc.ending_segment.end,
            })

        best_clip_data = candidates_data[0] if candidates_data else None

        # Stage 14: LLM teacher — label segments and candidates
        llm_labels = []
        if self.cfg.llm.USE_LLM:
            from .llm.teacher import LLMTeacher
            teacher = LLMTeacher()
            for seg in atomic_segments[:self.cfg.llm.MAX_SEGMENTS_TO_LABEL]:
                seg_label = self.run_stage("llm_teacher", teacher.label_segment, seg)
                if seg_label:
                    llm_labels.append(seg_label.to_dict())
            if best_scored:
                cand_label = self.run_stage("llm_teacher",
                                             teacher.label_candidate, best_scored.clip)
                if cand_label:
                    llm_labels.append(cand_label.to_dict())
            for sc in rejected_clips[:self.cfg.llm.MAX_REJECTED_TO_LABEL]:
                rlabel = self.run_stage("llm_teacher",
                                         teacher.label_candidate, sc.clip)
                if rlabel:
                    llm_labels.append(rlabel.to_dict())

        # Stage 15-16: Decision logging + failure storage
        if best_scored:
            self.decision_log.record(
                entity_id=best_scored.clip.hook_segment.id,
                entity_type="clip",
                stage="scoring",
                rule_name="total_score",
                rule_category="ranking",
                confidence=best_scored.total_score,
                outcome="selected",
            )
            self.tracker.record_candidate(
                candidate_id=f"{best_scored.clip.hook_segment.id}_{best_scored.clip.ending_segment.id}",
                score=best_scored.total_score,
                hook_id=best_scored.clip.hook_segment.id,
                body_ids=[s.id for s in best_scored.clip.body_segments],
                ending_id=best_scored.clip.ending_segment.id,
                accepted=True,
            )

        # Stages 17-19: Knowledge graph updates
        for seg in atomic_segments:
            self.context_db.register_segment(seg.id, seg.text)
            for pt in seg.patterns:
                self.global_graph.record_node(pt, position=seg.start / max(total_duration, 1))

        # Stages 20-24: Pattern intelligence
        pattern_result = self.pattern_engine.process_video(
            atomic_segments, category=category
        )

        # Stage 25: Confidence scoring
        final_confidence = self.confidence_propagator.propagate({
            "transcription": 0.98,
            "segmentation": 0.97,
            "feature_extraction": 0.97,
            "knowledge_graph": 0.96,
            "rule_engine": 0.95,
            "scoring": 0.94,
        })
        routing = self.threshold_engine.get_routing_details(final_confidence)

        # Save to SQLite
        self.storage.save_video(
            video_id=self.video_id,
            title=os.path.basename(video_path),
            duration=total_duration,
            audio_quality=f"snr={quality.snr_db:.1f}",
            language=raw_segments[0].get("language", "en") if raw_segments else "en",
            pipeline_version=self.cfg.storage.PIPELINE_VERSION,
        )
        for seg in atomic_segments:
            self.storage.save_segment(self.video_id, {
                "id": seg.id, "index": seg.index, "text": seg.text,
                "start": seg.start, "end": seg.end, "duration": seg.duration,
                "speaker": seg.speaker, "sentiment": seg.sentiment,
                "speech_rate": seg.speech_rate, "volume_delta": seg.volume_delta,
                "patterns": seg.patterns, "rules_matched": seg.rules_matched,
            })

        processing_time = time.time() - start_time
        self._stats.update({
            "total_duration": round(total_duration, 2),
            "processing_time_seconds": round(processing_time, 2),
            "pipeline_version": self.cfg.storage.PIPELINE_VERSION,
            "final_confidence": round(final_confidence, 3),
            "routing_decision": routing["action"],
            "patterns_discovered": pattern_result["video_patterns"],
        })

        return PipelineResult(
            video_id=self.video_id,
            candidates=candidates_data,
            best_clip=best_clip_data,
            stats=self._stats,
        )
