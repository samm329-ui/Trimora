from typing import Optional


ROUTING_MATRIX = [
    (0.90, 1.01, "USE_LOCAL_MODEL", "$0", "Instant"),
    (0.75, 0.90, "USE_PATTERN_MATCH", "$0", "~10ms"),
    (0.60, 0.75, "USE_RULE_ENGINE", "$0", "~5ms"),
    (0.30, 0.60, "USE_LLM", "~$0.001", "~2s"),
    (0.00, 0.30, "FLAG_FOR_HUMAN", "$0", "Manual"),
]


class AdaptiveThresholdEngine:
    def __init__(self):
        self._routing_history: list[dict] = []

    def get_routing_decision(self, confidence: float) -> str:
        for lo, hi, action, _, _ in ROUTING_MATRIX:
            if lo <= confidence < hi:
                return action
        return "USE_LLM"

    def get_routing_details(self, confidence: float) -> dict:
        for lo, hi, action, cost, speed in ROUTING_MATRIX:
            if lo <= confidence < hi:
                return {
                    "confidence": confidence,
                    "action": action,
                    "cost": cost,
                    "speed": speed,
                    "range": f"{lo:.2f}–{hi:.2f}",
                }
        return {
            "confidence": confidence,
            "action": "USE_LLM",
            "cost": "~$0.001",
            "speed": "~2s",
        }

    def should_use_llm(self, confidence: float) -> bool:
        return confidence < 0.6

    def should_flag_for_human(self, confidence: float) -> bool:
        return confidence < 0.3

    def record_routing(self, entity_id: str, decision_type: str,
                        confidence: float) -> None:
        decision = self.get_routing_details(confidence)
        self._routing_history.append({
            "entity_id": entity_id,
            "decision_type": decision_type,
            "confidence": confidence,
            "routing_decision": decision["action"],
        })

    def get_routing_summary(self) -> dict:
        counts: dict[str, int] = {}
        for entry in self._routing_history:
            routing = entry["routing_decision"]
            counts[routing] = counts.get(routing, 0) + 1
        return {
            "total_routings": len(self._routing_history),
            "routing_counts": counts,
        }

    def clear_history(self) -> None:
        self._routing_history.clear()
