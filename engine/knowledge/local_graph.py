from collections import defaultdict
from typing import Optional

import networkx as nx

from ..data.models import Segment


class LocalKnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self._segment_map: dict[str, Segment] = {}
        self._node_stats: dict[str, dict] = {}

    def add_segment(self, segment: Segment) -> None:
        self.graph.add_node(
            segment.id,
            segment=segment,
        )
        self._segment_map[segment.id] = segment
        self._node_stats[segment.id] = {
            "occurrences": 1,
            "emotion": "unknown",
            "avg_sentiment": segment.sentiment,
            "avg_duration": segment.duration,
        }

    def add_edge(self, source_id: str, target_id: str,
                 edge_type: str = "follows",
                 weight: float = 0.5,
                 confidence: float = 0.5) -> None:
        if self.graph.has_edge(source_id, target_id):
            existing = self.graph.edges[source_id, target_id]
            new_weight = max(existing.get("weight", 0), weight)
            self.graph.edges[source_id, target_id]["weight"] = new_weight
            self.graph.edges[source_id, target_id]["confidence"] = max(
                existing.get("confidence", 0), confidence
            )
            self.graph.edges[source_id, target_id]["occurrences"] = (
                existing.get("occurrences", 1) + 1
            )
        else:
            self.graph.add_edge(
                source_id, target_id,
                edge_type=edge_type,
                weight=weight,
                confidence=confidence,
                occurrences=1,
            )

    def get_segment(self, segment_id: str) -> Optional[Segment]:
        return self._segment_map.get(segment_id)

    def get_node_stats(self, segment_id: str) -> dict:
        return self._node_stats.get(segment_id, {})

    def get_in_edges(self, segment_id: str) -> list[dict]:
        return [
            {
                "source": u,
                "edge_type": d.get("edge_type", "unknown"),
                "weight": d.get("weight", 0),
                "confidence": d.get("confidence", 0),
            }
            for u, v, d in self.graph.in_edges(segment_id, data=True)
        ]

    def get_out_edges(self, segment_id: str) -> list[dict]:
        return [
            {
                "target": v,
                "edge_type": d.get("edge_type", "unknown"),
                "weight": d.get("weight", 0),
                "confidence": d.get("confidence", 0),
            }
            for u, v, d in self.graph.out_edges(segment_id, data=True)
        ]

    def get_temporal_neighbors(self, segment_id: str,
                                window_seconds: float = 60.0) -> list[Segment]:
        seg = self._segment_map.get(segment_id)
        if seg is None:
            return []
        return [
            s for s in self._segment_map.values()
            if s.id != segment_id
            and abs(s.start - seg.start) <= window_seconds
        ]

    def get_connected_cluster(self, segment_id: str,
                               max_duration: float = 90.0) -> list[Segment]:
        if segment_id not in self._segment_map:
            return []
        visited = set()
        queue = [segment_id]
        cluster = []
        total_dur = 0.0
        while queue and total_dur < max_duration:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            seg = self._segment_map.get(current)
            if seg is None:
                continue
            cluster.append(seg)
            total_dur += seg.duration
            for neighbor in self.graph.successors(current):
                if neighbor not in visited:
                    queue.append(neighbor)
            for neighbor in self.graph.predecessors(current):
                if neighbor not in visited:
                    queue.append(neighbor)
        return cluster

    def edge_count(self) -> int:
        return self.graph.number_of_edges()

    def node_count(self) -> int:
        return self.graph.number_of_nodes()

    def clear(self) -> None:
        self.graph.clear()
        self._segment_map.clear()
        self._node_stats.clear()
