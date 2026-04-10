"""MBS Search Strategy — Priority 6: Curriculum-aware exploration.

Implements Direction #6 (Search Strategy / Exploration-Exploitation). Replaces
the generic Draft/Improvement/Ensemble stage selection with MBS-specific
curriculum constraints.

Three mechanisms:

    1. Dependency gating: A hypothesis targeting component X is only allowed if
       its prerequisites have been met. For example, burnout hypotheses are
       blocked until `rate_incentive` is present AND the model shows monotonic
       rate sensitivity. Ensemble hypotheses are blocked until at least 3
       distinct successful architectures exist.

    2. Adaptive exploration radius: track improvement velocity. If the last N
       iterations each improved RMSE by >X%, stay in exploitation mode
       (refine current architecture). If improvement stalls, switch to
       exploration mode (try a different component/architecture family).

    3. Backtracking with cooldown: if the last K iterations failed to improve,
       revert to the best checkpoint and put the recently-failed component
       on a cooldown list so hypothesis_gen avoids it for a while.

The returned `HypothesisFilter` is passed into the hypothesis_select LLM call
as a constraint list — the LLM must propose only allowed components.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class ExplorationMode(str, Enum):
    EXPLOITATION = "exploitation"  # refine best model
    EXPLORATION = "exploration"    # try different architecture family
    BACKTRACK = "backtrack"        # revert to checkpoint + cooldown


class MBSComponent(str, Enum):
    DATA_LOADER = "DataLoader"
    RATE_CURVE_FEATURES = "RateCurveFeatures"
    POOL_DYNAMICS = "PoolDynamics"
    MACRO_FEATURES = "MacroFeatures"
    PREPAYMENT_MODEL = "PrepaymentModel"
    SCENARIO_VALIDATOR = "ScenarioValidator"
    ENSEMBLE = "Ensemble"


#: Prerequisites: component X requires all preconditions in this mapping to be
#: satisfied before hypotheses targeting X are allowed.
COMPONENT_PREREQUISITES: dict[str, list[str]] = {
    "RateCurveFeatures": [],
    "DataLoader": [],
    "PoolDynamics": ["rate_incentive_present", "rate_sensitivity_monotonic"],
    "MacroFeatures": ["rate_incentive_present"],
    "PrepaymentModel": ["rate_incentive_present"],
    "ScenarioValidator": ["rate_sensitivity_monotonic"],
    "Ensemble": ["n_distinct_architectures_ge_3"],
}


@dataclass
class IterationRecord:
    iteration: int
    component_touched: str
    overall_rmse: float
    rate_sensitivity_monotonic: bool
    has_rate_incentive: bool
    architecture_family: str
    success: bool
    economic_violations: int = 0


@dataclass
class MBSSearchState:
    """Persisted search state across iterations."""

    history: list[IterationRecord] = field(default_factory=list)
    cooldown: dict[str, int] = field(default_factory=dict)  # component → iterations remaining
    improvement_threshold: float = 0.05  # 5% relative RMSE improvement = "meaningful"
    stall_window: int = 3
    cooldown_duration: int = 2
    backtrack_trigger: int = 3  # N consecutive failures triggers backtrack

    def append(self, record: IterationRecord) -> None:
        self.history.append(record)
        # Tick down all cooldowns
        self.cooldown = {
            k: v - 1 for k, v in self.cooldown.items() if v - 1 > 0
        }

    def best_rmse(self) -> float:
        successes = [r for r in self.history if r.success]
        return min((r.overall_rmse for r in successes), default=float("inf"))

    def last_k_improvements(self, k: int) -> list[float]:
        """Relative RMSE improvements over last k iterations."""
        hist = [r for r in self.history if r.success]
        if len(hist) < 2:
            return []
        deltas: list[float] = []
        for i in range(max(1, len(hist) - k), len(hist)):
            prev = hist[i - 1].overall_rmse
            cur = hist[i].overall_rmse
            if prev > 1e-12:
                deltas.append((prev - cur) / prev)
        return deltas

    def consecutive_failures(self) -> int:
        count = 0
        for r in reversed(self.history):
            if r.success and r.overall_rmse <= self.best_rmse():
                break
            count += 1
        return count

    def distinct_architectures(self) -> int:
        return len({r.architecture_family for r in self.history if r.success})

    def has_rate_incentive(self) -> bool:
        return any(r.has_rate_incentive for r in self.history if r.success)

    def rate_sensitivity_monotonic(self) -> bool:
        recent_successes = [r for r in self.history if r.success]
        if not recent_successes:
            return False
        return recent_successes[-1].rate_sensitivity_monotonic

    # --- Prerequisite evaluation -----------------------------------------

    def check_prerequisite(self, prereq: str) -> bool:
        if prereq == "rate_incentive_present":
            return self.has_rate_incentive()
        if prereq == "rate_sensitivity_monotonic":
            return self.rate_sensitivity_monotonic()
        if prereq == "n_distinct_architectures_ge_3":
            return self.distinct_architectures() >= 3
        return False

    # --- Persistence -----------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "history": [
                {
                    "iteration": r.iteration,
                    "component_touched": r.component_touched,
                    "overall_rmse": r.overall_rmse,
                    "rate_sensitivity_monotonic": r.rate_sensitivity_monotonic,
                    "has_rate_incentive": r.has_rate_incentive,
                    "architecture_family": r.architecture_family,
                    "success": r.success,
                    "economic_violations": r.economic_violations,
                }
                for r in self.history
            ],
            "cooldown": self.cooldown,
        }
        path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> MBSSearchState:
        path = Path(path)
        if not path.exists():
            return cls()
        data = json.loads(path.read_text())
        state = cls(cooldown=data.get("cooldown", {}))
        for r in data.get("history", []):
            state.history.append(IterationRecord(**r))
        return state


@dataclass
class HypothesisFilter:
    """The constraint object passed to hypothesis_select."""

    allowed_components: list[str]
    blocked_components: dict[str, str]  # component → reason
    mode: ExplorationMode
    guidance: str
    should_backtrack: bool = False


def decide_next_iteration(state: MBSSearchState) -> HypothesisFilter:
    """Produce the curriculum constraint for the next hypothesis_select call."""
    allowed: list[str] = []
    blocked: dict[str, str] = {}

    # 1. Dependency gating
    for component, prereqs in COMPONENT_PREREQUISITES.items():
        if component in state.cooldown and state.cooldown[component] > 0:
            blocked[component] = (
                f"Component on cooldown for {state.cooldown[component]} more iterations "
                f"after recent failures"
            )
            continue
        unmet = [p for p in prereqs if not state.check_prerequisite(p)]
        if unmet:
            blocked[component] = f"Prerequisites not met: {', '.join(unmet)}"
        else:
            allowed.append(component)

    # 2. Adaptive exploration mode based on improvement velocity
    deltas = state.last_k_improvements(state.stall_window)
    if not deltas:
        mode = ExplorationMode.EXPLORATION
        guidance = (
            "No successful iterations yet — explore broadly. Propose a baseline "
            "that establishes the temporal train/test split, `rate_incentive` "
            "feature, and a simple model (Ridge or small GBM)."
        )
    elif all(d >= state.improvement_threshold for d in deltas):
        mode = ExplorationMode.EXPLOITATION
        guidance = (
            f"Last {len(deltas)} iterations all improved RMSE by ≥{state.improvement_threshold:.0%}. "
            "Stay in exploitation mode: refine the current architecture with incremental "
            "feature additions or hyperparameter tuning within the same model family."
        )
    elif all(d < state.improvement_threshold for d in deltas):
        mode = ExplorationMode.EXPLORATION
        guidance = (
            f"Last {len(deltas)} iterations all improved RMSE by <{state.improvement_threshold:.0%}. "
            "Progress has stalled — switch to exploration. Propose a hypothesis "
            "targeting a different component branch or a different model family "
            "(e.g., from GBM to neural net with sigmoid output, or from unified "
            "to decomposed turnover + refi model)."
        )
    else:
        mode = ExplorationMode.EXPLOITATION
        guidance = "Mixed improvement signal — continue exploitation with small architectural variations."

    # 3. Backtrack trigger
    failures = state.consecutive_failures()
    should_backtrack = failures >= state.backtrack_trigger
    if should_backtrack:
        mode = ExplorationMode.BACKTRACK
        # Put the most-touched recent component on cooldown
        recent = [r.component_touched for r in state.history[-state.backtrack_trigger:]]
        if recent:
            # Pick most common
            most = max(set(recent), key=recent.count)
            state.cooldown[most] = state.cooldown_duration
            blocked[most] = f"Cooldown after {failures} consecutive failures"
            allowed = [c for c in allowed if c != most]
        guidance = (
            f"{failures} consecutive iterations failed to improve over SOTA. "
            f"Reverting to the best checkpoint. The recently-failed component "
            f"is on cooldown for {state.cooldown_duration} iterations. "
            "Propose a hypothesis targeting a different component branch."
        )

    return HypothesisFilter(
        allowed_components=allowed,
        blocked_components=blocked,
        mode=mode,
        guidance=guidance,
        should_backtrack=should_backtrack,
    )


def format_filter_for_prompt(filt: HypothesisFilter) -> str:
    """Render the filter as a Markdown block for injection into hypothesis_gen."""
    lines = [
        "## Search Strategy Constraints (enforce these in your hypothesis)",
        f"**Mode**: {filt.mode.value}",
        f"**Allowed components**: {', '.join(filt.allowed_components) if filt.allowed_components else '(none — propose a baseline)'}",
    ]
    if filt.blocked_components:
        lines.append("**Blocked components**:")
        for comp, reason in filt.blocked_components.items():
            lines.append(f"  - {comp}: {reason}")
    lines.append(f"\n**Guidance**: {filt.guidance}")
    if filt.should_backtrack:
        lines.append("\n**⚠ BACKTRACK**: Revert to the best checkpoint before proposing.")
    return "\n".join(lines)
