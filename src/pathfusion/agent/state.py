"""Agent state shared between the loop and the proposers."""

from __future__ import annotations

from dataclasses import dataclass, field

from pathfusion.agent.evaluator import TrialResult
from pathfusion.fusion.spec import FusionSpec


@dataclass
class TrialRecord:
    index: int                      # 0-based over all evaluated specs (references included)
    phase: str                      # reference | coverage | adaptive
    source: str                     # fixed | deterministic | llm:<name> | fallback
    reasoning: str
    result: TrialResult
    diff_vs_incumbent: float | None = None
    se_diff: float | None = None
    accepted: bool = False
    incumbent_after: str = ""

    @property
    def spec(self) -> FusionSpec:
        return self.result.spec


@dataclass
class AgentState:
    budget: int
    records: list[TrialRecord] = field(default_factory=list)
    incumbent: TrialRecord | None = None
    best_reference: TrialRecord | None = None
    since_improvement: int = 0

    @property
    def fusion_records(self) -> list[TrialRecord]:
        return [r for r in self.records if r.phase != "reference"]

    @property
    def trials_used(self) -> int:
        return len(self.fusion_records)

    @property
    def remaining(self) -> int:
        return self.budget - self.trials_used

    def tried(self) -> set[str]:
        return {r.spec.key for r in self.records}
