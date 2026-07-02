import sys, os, json

os.environ["GROQ_API_KEY"] = "test-key"

from engine.config import get_config
cfg = get_config()
cfg.llm.USE_LLM = False

from engine.pipeline import Pipeline

p = Pipeline(video_id="smoke-test-001")

print("=== Pipeline Initialized ===")
print(f"  video_id: {p.video_id}")
print(f"  RESUME_ENABLED: {cfg.pipeline.RESUME_ENABLED}")
print(f"  PIPELINE_VERSION: {cfg.storage.PIPELINE_VERSION}")
print()

completed = p.get_completed_stages()
print(f"  completed_stages (fresh): {completed}")
p.mark_stage_complete("test_stage")
completed = p.get_completed_stages()
print(f"  completed_stages (after mark): {completed}")
print()

from engine.pipeline import PipelineError
def failing_fn():
    raise PipelineError("test", "test error", recoverable=True)
result = p.run_stage("test", failing_fn)
print(f"  run_stage recoverable: result={result}")
print()

from engine.decision.log import DecisionEntry
entry = DecisionEntry(
    entity_id="seg_1", entity_type="segment", stage="test",
    rule_name="test_rule", rule_category="test", confidence=0.9,
    outcome="selected",
)
print(f"  DecisionEntry: {entry.to_dict()}")
print()

from engine.decision.failures import FailureStore
from engine.scoring.scorer import ScoredClip, ValidClip
from engine.data.models import Segment
seg = Segment(id="seg_1", text="test", start=0.0, end=10.0, duration=10.0)
seg2 = Segment(id="seg_2", text="test2", start=10.0, end=20.0, duration=10.0)
seg3 = Segment(id="seg_3", text="test3", start=20.0, end=30.0, duration=10.0)
vc = ValidClip(
    hook_segment=seg, body_segments=[seg2], ending_segment=seg3,
    total_duration=30.0, hook_duration=10.0,
)
sc = ScoredClip(clip=vc, total_score=0.5)
fs = FailureStore(video_id="test")
fs.record(sc, rules_failed=["test_rule"], rules_passed=[])
fs.record_decision(accepted=False)
print(f"  failure rate: {fs.get_failure_rate()}")
print()

from engine.confidence.scorer import ConfidencePropagator
cp = ConfidencePropagator()
conf_result = cp.propagate({"transcription": 0.98, "segmentation": 0.97})
print(f"  confidence propagation: {conf_result:.4f}")
print()

from engine.confidence.threshold import AdaptiveThresholdEngine
ate = AdaptiveThresholdEngine()
print(f"  routing@0.95: {ate.get_routing_decision(0.95)}")
print(f"  routing@0.70: {ate.get_routing_decision(0.70)}")
print(f"  routing@0.40: {ate.get_routing_decision(0.40)}")
print()

from engine.knowledge.global_graph import GlobalKnowledgeGraph
gg = GlobalKnowledgeGraph()
gg.record_edge("curiosity_hook", "personal_story", watch_time_pct=0.74, saves=68, shares=55, llm_confidence=0.91)
gg.record_edge("statistics", "framework", watch_time_pct=0.61, saves=82, shares=43, llm_confidence=0.87)
gg.record_node("curiosity_hook", position=4.2, sentiment=0.65, duration=4.8, emotion="curiosity")
print(f"  global graph edges: {gg.edge_count()}")
print(f"  global graph nodes: {gg.node_count()}")
summary = gg.summary()
print(f"  global graph summary: edges={summary['total_edge_types']}, nodes={summary['total_node_types']}")
print()

from engine.data.local_store import LocalStore
ls = LocalStore()
ls.save_json("test-vid", "test.json", {"key": "value"})
loaded = ls.load_json("test-vid", "test.json")
print(f"  LocalStore save/load: {loaded}")
print()

from engine.data.storage import Storage
s = Storage(db_path="engine/data/store/test_trimora.db")
s.save_video(video_id="test-vid", title="test", duration=30.0)
vid = s.get_video("test-vid")
print(f"  SQLite save/get video: {vid['video_id'] if vid else 'NOT FOUND'}")
s.close()
print()

from engine.data.migrations import MigrationRunner
mr = MigrationRunner(db_path="engine/data/store/test_trimora.db")
applied = mr.run_pending()
print(f"  migrations applied: {applied}")
print()

from engine.patterns.engine import PatternIntelligenceEngine
pie = PatternIntelligenceEngine()
segments = [seg, seg2, seg3]
pattern_result = pie.process_video(segments, category="test")
print(f"  pattern engine: {pattern_result['video_patterns']} discovered, {pattern_result['global_matches']} matches")
print()

from engine.patterns.graph import PatternVersion, PatternEvolution
pe = PatternEvolution()
v1 = pe.create_version(421, ["Question", "Story", "Failure", "Lesson"], trigger="discovered")
print(f"  PatternVersion v1: {v1.version}, status={v1.status}")
from engine.patterns.detector import DiscoveredPattern
variant = DiscoveredPattern(
    node_types=["Question", "Story", "Failure", "Framework", "Lesson"],
)
v2 = pe.track_variant(421, variant)
print(f"  PatternEvolution variant: version={v2.version}, trigger={v2.evolution_trigger}")
versions = pe.get_versions(421)
print(f"  versions count: {len(versions)}")
print()

from engine.patterns.embeddings import PatternEmbeddings
emb = PatternEmbeddings()
emb.add_embedding("Q->S->F->L", [0.42, 0.87, 0.13, 0.56])
similar = emb.find_similar([0.41, 0.85, 0.15, 0.54])
print(f"  embeddings similar: {similar}")
print()

print("=== ALL SMOKE TESTS PASSED ===")
