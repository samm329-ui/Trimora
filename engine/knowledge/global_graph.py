import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional


class GlobalKnowledgeGraph:
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path
        self.edges: dict[tuple[str, str], dict] = {}
        self.nodes: dict[str, dict] = {}

    def record_edge(self, source_type: str, target_type: str,
                     watch_time_pct: float = 0.0, saves: int = 0,
                     shares: int = 0, llm_confidence: float = 0.0) -> None:
        key = (source_type, target_type)
        now = datetime.now(timezone.utc).isoformat()
        if key in self.edges:
            e = self.edges[key]
            e["occurrences"] += 1
            n = e["occurrences"]
            e["avg_watch_time"] = (e["avg_watch_time"] * (n - 1) + watch_time_pct) / n
            e["avg_saves"] = (e["avg_saves"] * (n - 1) + saves) / n
            e["avg_shares"] = (e["avg_shares"] * (n - 1) + shares) / n
            e["llm_confidence"] = max(e["llm_confidence"], llm_confidence)
            e["avg_confidence"] = (e["avg_confidence"] * (n - 1) + llm_confidence) / n
            e["last_seen"] = now
        else:
            self.edges[key] = {
                "source_type": source_type,
                "target_type": target_type,
                "occurrences": 1,
                "avg_watch_time": watch_time_pct,
                "avg_saves": saves,
                "avg_shares": shares,
                "llm_confidence": llm_confidence,
                "avg_confidence": llm_confidence,
                "first_seen": now,
                "last_seen": now,
            }

    def record_node(self, node_type: str, position: float = 0.0,
                     sentiment: float = 0.0, duration: float = 0.0,
                     emotion: str = "") -> None:
        now = datetime.now(timezone.utc).isoformat()
        if node_type in self.nodes:
            n = self.nodes[node_type]
            n["occurrences"] += 1
            oc = n["occurrences"]
            n["avg_position"] = (n["avg_position"] * (oc - 1) + position) / oc
            n["avg_sentiment"] = (n["avg_sentiment"] * (oc - 1) + sentiment) / oc
            n["avg_duration"] = (n["avg_duration"] * (oc - 1) + duration) / oc
            n["last_seen"] = now
            if emotion and emotion != "unknown":
                n["emotion_counts"][emotion] = n["emotion_counts"].get(emotion, 0) + 1
        else:
            self.nodes[node_type] = {
                "node_type": node_type,
                "occurrences": 1,
                "avg_position": position,
                "avg_sentiment": sentiment,
                "avg_duration": duration,
                "emotion_counts": {emotion: 1} if emotion else {},
                "first_seen": now,
                "last_seen": now,
            }

    def get_edge(self, source_type: str, target_type: str) -> Optional[dict]:
        return self.edges.get((source_type, target_type))

    def get_node(self, node_type: str) -> Optional[dict]:
        return self.nodes.get(node_type)

    def get_followers(self, source_type: str, top_n: int = 5) -> list[dict]:
        results = []
        for (src, tgt), data in self.edges.items():
            if src == source_type:
                results.append(data)
        results.sort(key=lambda x: x["occurrences"], reverse=True)
        return results[:top_n]

    def get_predecessors(self, target_type: str, top_n: int = 5) -> list[dict]:
        results = []
        for (src, tgt), data in self.edges.items():
            if tgt == target_type:
                results.append(data)
        results.sort(key=lambda x: x["occurrences"], reverse=True)
        return results[:top_n]

    def get_top_edges(self, n: int = 10) -> list[dict]:
        return sorted(self.edges.values(),
                      key=lambda x: x["occurrences"], reverse=True)[:n]

    def get_top_nodes(self, n: int = 10) -> list[dict]:
        return sorted(self.nodes.values(),
                      key=lambda x: x["occurrences"], reverse=True)[:n]

    def get_dominant_emotion(self, node_type: str) -> str:
        n = self.nodes.get(node_type)
        if not n or not n["emotion_counts"]:
            return "unknown"
        return max(n["emotion_counts"], key=n["emotion_counts"].get)

    def edge_count(self) -> int:
        return len(self.edges)

    def node_count(self) -> int:
        return len(self.nodes)

    def summary(self) -> dict:
        return {
            "total_edge_types": self.edge_count(),
            "total_node_types": self.node_count(),
            "top_edges": self.get_top_edges(5),
            "top_nodes": self.get_top_nodes(5),
        }

    def save(self, filepath: Optional[str] = None) -> None:
        path = filepath or self.storage_path
        if path:
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump({
                    "edges": {f"{k[0]}->{k[1]}": v for k, v in self.edges.items()},
                    "nodes": self.nodes,
                }, f, indent=2, default=str)

    def load(self, filepath: Optional[str] = None) -> None:
        path = filepath or self.storage_path
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for key_str, value in data.get("edges", {}).items():
                parts = key_str.split("->", 1)
                if len(parts) == 2:
                    self.edges[(parts[0], parts[1])] = value
            self.nodes.update(data.get("nodes", {}))
