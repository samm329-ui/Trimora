from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from .detector import DiscoveredPattern


@dataclass
class PatternVersion:
    pattern_id: int
    version: str
    parent_version: str
    node_sequence: list[str]
    occurrences: int = 0
    avg_saves: float = 0.0
    avg_shares: float = 0.0
    confidence: float = 0.5
    confidence_history: list[dict] = field(default_factory=list)
    status: str = "ACTIVE"
    created_at: str = ""
    superseded_at: str = ""
    evolution_trigger: str = "new_data"

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def compare_performance(self, other: 'PatternVersion') -> dict:
        return {
            "saves_delta": self.avg_saves - other.avg_saves,
            "shares_delta": self.avg_shares - other.avg_shares,
            "confidence_delta": self.confidence - other.confidence,
            "sample_size_ratio": self.occurrences / max(other.occurrences, 1),
        }


class PatternEvolution:
    def __init__(self):
        self._versions: dict[str, list[PatternVersion]] = {}
        self._version_counter: dict[int, int] = {}
        self._log: list[dict] = []

    def create_version(self, parent_id: int, node_sequence: list[str],
                        trigger: str = "new_data") -> PatternVersion:
        counter = self._version_counter.get(parent_id, 0) + 1
        self._version_counter[parent_id] = counter
        major = parent_id
        minor = counter
        version_str = f"{major}.{minor}"
        parent_version = f"{major}.{counter - 1}" if counter > 1 else ""
        version = PatternVersion(
            pattern_id=parent_id,
            version=version_str,
            parent_version=parent_version,
            node_sequence=node_sequence,
            evolution_trigger=trigger,
        )
        key = str(parent_id)
        if key not in self._versions:
            self._versions[key] = []
        self._versions[key].append(version)
        return version

    def promote(self, version_id: str) -> None:
        for versions in self._versions.values():
            for v in versions:
                if v.version == version_id:
                    v.status = "ACTIVE"
                elif v.pattern_id == versions[0].pattern_id:
                    if v.status == "ACTIVE":
                        v.status = "SUPERSEDED"
                        v.superseded_at = datetime.now(timezone.utc).isoformat()

    def track_variant(self, parent_id: int, variant: DiscoveredPattern) -> PatternVersion:
        parent_versions = self._versions.get(str(parent_id), [])
        parent = parent_versions[-1] if parent_versions else None
        delta_saves = 0.0
        if parent:
            delta_saves = variant.avg_saves - parent.avg_saves
        trigger = "improved_performance" if delta_saves > 0.05 else "new_data"
        new_version = self.create_version(
            parent_id=parent_id,
            node_sequence=variant.node_types,
            trigger=trigger,
        )
        if delta_saves > 0.05:
            self.promote(new_version.version)
        self._log.append({
            "parent_id": parent_id,
            "variant_id": id(variant),
            "version": new_version.version,
            "delta_saves": delta_saves,
            "trigger": trigger,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        return new_version

    def get_versions(self, pattern_id: int) -> list[PatternVersion]:
        return self._versions.get(str(pattern_id), [])

    def evolution_log(self) -> list[dict]:
        return list(self._log)


class PatternGraph:
    def __init__(self):
        self._patterns: dict[str, DiscoveredPattern] = {}
        self._edges: dict[tuple[str, str], dict] = {}
        self._follower_cache: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def add_pattern(self, pattern: DiscoveredPattern) -> None:
        key = "->".join(pattern.node_types)
        self._patterns[key] = pattern

        for i in range(len(pattern.node_types) - 1):
            src = pattern.node_types[i]
            tgt = pattern.node_types[i + 1]
            self._follower_cache[src][tgt] += 1

            edge_key = (src, tgt)
            if edge_key in self._edges:
                self._edges[edge_key]["occurrences"] += 1
            else:
                self._edges[edge_key] = {
                    "source": src,
                    "target": tgt,
                    "occurrences": 1,
                }

    def get_follower_distribution(self, node_type: str) -> dict[str, int]:
        return dict(self._follower_cache.get(node_type, {}))

    def get_pattern(self, node_types: list[str]) -> Optional[DiscoveredPattern]:
        key = "->".join(node_types)
        return self._patterns.get(key)

    def get_edge(self, source: str, target: str) -> Optional[dict]:
        return self._edges.get((source, target))

    def pattern_count(self) -> int:
        return len(self._patterns)

    def edge_count(self) -> int:
        return len(self._edges)

    def top_patterns(self, n: int = 10) -> list[DiscoveredPattern]:
        return sorted(
            self._patterns.values(),
            key=lambda p: p.occurrences,
            reverse=True,
        )[:n]
