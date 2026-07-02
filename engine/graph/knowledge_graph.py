import networkx as nx
from ..data.models import Segment
from ..config import get_config


class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_segment(self, segment: Segment):
        self.graph.add_node(
            segment.id,
            segment=segment,
            text=segment.text,
            start=segment.start,
            end=segment.end,
            duration=segment.duration,
            sentiment=segment.sentiment,
            patterns=segment.patterns,
            position=segment.start / max(segment.end, 1)
        )

    def add_relationship(self, source_id: str, target_id: str,
                         edge_type: str, weight: float, evidence: str):
        if not self.graph.has_node(source_id) or not self.graph.has_node(target_id):
            return
        if self.graph.has_edge(source_id, target_id):
            existing = self.graph.edges[source_id, target_id]
            if weight > existing.get("weight", 0):
                self.graph.add_edge(
                    source_id, target_id,
                    type=edge_type, weight=weight, evidence=evidence
                )
        else:
            self.graph.add_edge(
                source_id, target_id,
                type=edge_type, weight=weight, evidence=evidence
            )

    def get_temporal_neighbors(self, segment_id: str, window: int = 3) -> list[str]:
        if not self.graph.has_node(segment_id):
            return []

        node_data = self.graph.nodes[segment_id]
        seg_start = node_data.get("start", 0)

        sorted_nodes = sorted(
            [(n, self.graph.nodes[n].get("start", 0)) for n in self.graph.nodes()],
            key=lambda x: x[1]
        )

        own_index = None
        for i, (nid, _) in enumerate(sorted_nodes):
            if nid == segment_id:
                own_index = i
                break

        if own_index is None:
            return []

        start = max(0, own_index - window)
        end = min(len(sorted_nodes), own_index + window + 1)

        return [sorted_nodes[i][0] for i in range(start, end) if sorted_nodes[i][0] != segment_id]

    def get_connected_cluster(self, start_id: str, max_duration: float = None) -> list[str]:
        if not self.graph.has_node(start_id):
            return []

        if max_duration is None:
            max_duration = get_config().graph.BFS_MAX_DURATION

        visited = set()
        queue = [(start_id, 0.0)]
        cluster = []

        while queue:
            node_id, accum = queue.pop(0)
            if node_id in visited or accum > max_duration:
                continue
            visited.add(node_id)
            cluster.append(node_id)

            for neighbor in self.graph.successors(node_id):
                seg = self.graph.nodes[neighbor].get("segment")
                if seg is None:
                    queue.append((neighbor, accum))
                else:
                    new_accum = accum + seg.duration
                    if new_accum <= max_duration:
                        queue.append((neighbor, new_accum))

        return cluster

    def get_segment(self, segment_id: str) -> Segment:
        if self.graph.has_node(segment_id):
            return self.graph.nodes[segment_id].get("segment")
        return None

    def node_count(self) -> int:
        return self.graph.number_of_nodes()

    def edge_count(self) -> int:
        return self.graph.number_of_edges()
