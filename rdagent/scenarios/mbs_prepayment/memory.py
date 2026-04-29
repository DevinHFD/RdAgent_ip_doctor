"""MBS Memory / Context Management — Priority 7.

Implements Direction #2 (Context Window and Memory Management). Replaces the
generic "last 10 experiments" trace with a structured per-iteration memory that:

    1. Extracts and persists KEY MODEL PROPERTIES (not raw code) after each
       successful experiment: rate sensitivity slope, burnout half-life,
       per-coupon RMSE, seasonal amplitude, etc. The LLM reasons much better
       over "burnout halflife is 8 months, literature suggests 12–18" than
       over reading 400 lines of model code.

    2. Groups trace entries by component (turnover, refi, curtailment,
       ensemble, feature-eng) — storing the BEST experiment per component
       plus the 3 most recent failures — instead of chronological last-10.
       This gives the LLM structural memory of what's been tried per
       component.

    3. Compresses code diffs to mathematical-change summaries:
       "Changed refi response from linear(rate_incentive) to
       sigmoid(rate_incentive, steepness=2.5)" rather than full diff.

    4. Implements phase-dependent context loading so only the fields
       relevant to the current iteration phase are loaded into the prompt:
         - hypothesis_gen: structured properties + attributions + last 3 feedbacks
         - coding: SOTA code + spec (not experiment history)
         - feedback: SOTA results + current results + diff (not RAG)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any


class IterationPhase(str, Enum):
    """Phases of a single RD-Agent iteration, each needing different context."""
    HYPOTHESIS_GEN = "hypothesis_gen"
    CODING = "coding"
    FEEDBACK = "feedback"


# ---------------------------------------------------------------------------
# Structured model properties extracted per iteration
# ---------------------------------------------------------------------------


@dataclass
class ModelProperties:
    """Compact structured summary of an experiment — one line per property.

    Extracted from the MBSEvaluationHarness scorecard and the fitted model
    metadata. This is what the next iteration's hypothesis_gen reads, NOT
    the raw code.
    """
    iteration: int
    model_type: str
    component_touched: str
    overall_rmse: float
    rmse_by_coupon_bucket: dict[str, float] = field(default_factory=dict)
    # S-curve bin RMSE between UPB-weighted actual and predicted SMM bin
    # curves over Avg_Prop_Refi_Incentive_WAC_30yr_2mos. Smaller is better.
    s_curve_rmse_overall: float = float("nan")
    s_curve_rmse_mid_belly: float = float("nan")
    burnout_halflife_months: float | None = None
    seasonal_amplitude: float = 0.0
    holdout_rmse: float = 0.0
    training_rmse: float | None = None
    n_features_used: int = 0
    cusip_differentiation_std: float = 0.0
    regime_transition_rmse_mean: float = 0.0

    def summary_line(self) -> str:
        """One-line description suitable for a trace table."""
        return (
            f"iter={self.iteration} model={self.model_type} "
            f"component={self.component_touched} "
            f"rmse={self.overall_rmse:.5f} "
            f"s_curve_mid={self.s_curve_rmse_mid_belly:.5f}"
        )

    @classmethod
    def from_scorecard(
        cls,
        iteration: int,
        model_type: str,
        component_touched: str,
        scorecard: dict[str, Any],
        n_features_used: int = 0,
        training_rmse: float | None = None,
        burnout_halflife_months: float | None = None,
    ) -> ModelProperties:
        acc = scorecard.get("accuracy", {})
        rs = scorecard.get("rate_sensitivity", {})
        tr = scorecard.get("temporal_robustness", {})
        sp = scorecard.get("structural_properties", {})
        regime = tr.get("regime_transition_rmse", {}) or {}
        regime_mean = (
            float(sum(regime.values()) / max(len(regime), 1)) if regime else 0.0
        )
        return cls(
            iteration=iteration,
            model_type=model_type,
            component_touched=component_touched,
            overall_rmse=float(acc.get("overall_rmse", 0.0)),
            rmse_by_coupon_bucket={
                k: float(v) for k, v in (acc.get("rmse_by_coupon_bucket", {}) or {}).items()
            },
            s_curve_rmse_overall=float(rs.get("s_curve_rmse_overall", float("nan"))),
            s_curve_rmse_mid_belly=float(rs.get("s_curve_rmse_mid_belly", float("nan"))),
            burnout_halflife_months=burnout_halflife_months,
            seasonal_amplitude=float(sp.get("seasonality_residual_range", 0.0) or 0.0),
            holdout_rmse=float(acc.get("overall_rmse", 0.0)),
            training_rmse=training_rmse,
            n_features_used=n_features_used,
            cusip_differentiation_std=float(sp.get("cusip_differentiation_std", 0.0) or 0.0),
            regime_transition_rmse_mean=regime_mean,
        )


# ---------------------------------------------------------------------------
# Trace compression: component-grouped storage with code-diff summaries
# ---------------------------------------------------------------------------


@dataclass
class TraceEntry:
    iteration: int
    component: str
    hypothesis: str
    code_change_summary: str   # mathematical-change summary, NOT a diff
    decision: str              # "accept" | "reject"
    success: bool
    properties: ModelProperties | None = None
    feedback_reason: str = ""


@dataclass
class MBSMemory:
    """Persistent memory across iterations with phase-aware context loading."""

    memory_path: Path
    max_failures_shown: int = 3
    max_properties_shown: int = 5

    def __post_init__(self):
        self.memory_path = Path(self.memory_path)
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.memory_path.exists():
            return {"entries": [], "properties": []}
        try:
            return json.loads(self.memory_path.read_text())
        except json.JSONDecodeError:
            return {"entries": [], "properties": []}

    def _save(self) -> None:
        self.memory_path.write_text(json.dumps(self._data, indent=2, default=str))

    # --- Append -----------------------------------------------------------

    def append_entry(self, entry: TraceEntry) -> None:
        rec = {
            "iteration": entry.iteration,
            "component": entry.component,
            "hypothesis": entry.hypothesis,
            "code_change_summary": entry.code_change_summary,
            "decision": entry.decision,
            "success": entry.success,
            "feedback_reason": entry.feedback_reason,
            "properties": asdict(entry.properties) if entry.properties else None,
        }
        self._data["entries"].append(rec)
        if entry.success and entry.properties is not None:
            self._data["properties"].append(asdict(entry.properties))
        self._save()

    # --- Query: best per component ---------------------------------------

    def best_per_component(self) -> dict[str, dict[str, Any]]:
        """Return best (lowest overall_rmse) successful entry per component."""
        best: dict[str, dict[str, Any]] = {}
        for e in self._data["entries"]:
            if not e.get("success") or not e.get("properties"):
                continue
            comp = e["component"]
            rmse = e["properties"]["overall_rmse"]
            if comp not in best or rmse < best[comp]["properties"]["overall_rmse"]:
                best[comp] = e
        return best

    def recent_failures(self, k: int | None = None) -> list[dict[str, Any]]:
        k = k or self.max_failures_shown
        failures = [e for e in self._data["entries"] if not e.get("success")]
        return failures[-k:]

    def latest_properties(self, k: int | None = None) -> list[dict[str, Any]]:
        k = k or self.max_properties_shown
        return self._data.get("properties", [])[-k:]

    # --- Phase-dependent context rendering -------------------------------

    def render_context(self, phase: IterationPhase) -> str:
        """Return the Markdown block appropriate for the given iteration phase."""
        if phase == IterationPhase.HYPOTHESIS_GEN:
            return self._render_hypothesis_context()
        if phase == IterationPhase.CODING:
            return self._render_coding_context()
        if phase == IterationPhase.FEEDBACK:
            return self._render_feedback_context()
        return ""

    def _render_hypothesis_context(self) -> str:
        """Structured model properties + recent failures — NO full code."""
        lines = ["## Memory: Experiment Context (hypothesis phase)"]
        best = self.best_per_component()
        if best:
            lines.append("\n### Best successful experiment per component")
            for comp, e in best.items():
                p = e["properties"]
                mid = p.get("s_curve_rmse_mid_belly", float("nan"))
                mid_str = f"{mid:.5f}" if mid == mid else "n/a"
                lines.append(
                    f"- **{comp}** (iter {e['iteration']}): rmse={p['overall_rmse']:.5f}, "
                    f"s_curve_rmse_mid_belly={mid_str}"
                )
                if p.get("rmse_by_coupon_bucket"):
                    coupon_str = ", ".join(
                        f"{k}:{v:.5f}" for k, v in p["rmse_by_coupon_bucket"].items()
                    )
                    lines.append(f"    per-coupon RMSE: {coupon_str}")
                lines.append(f"    hypothesis: {e['hypothesis'][:160]}")

        failures = self.recent_failures()
        if failures:
            lines.append(f"\n### Most recent {len(failures)} failures")
            for f in failures:
                lines.append(
                    f"- iter {f['iteration']} ({f['component']}): "
                    f"{f.get('feedback_reason', '(no reason given)')[:200]}"
                )

        latest = self.latest_properties()
        if latest:
            lines.append("\n### Latest model properties (trend)")
            for p in latest:
                lines.append(
                    f"- iter {p['iteration']}: rmse={p['overall_rmse']:.5f}, "
                    f"cusip_diff_std={p['cusip_differentiation_std']:.4f}"
                )
        return "\n".join(lines)

    def _render_coding_context(self) -> str:
        """Minimal — coder needs full SOTA code, not experiment history."""
        lines = [
            "## Memory: Coding Phase Context",
            "(experiment history omitted — coder receives full SOTA code and spec separately)",
        ]
        # Still useful: last successful code_change_summary for the same component
        best = self.best_per_component()
        if best:
            lines.append("\n### Recent accepted code-change summaries")
            for comp, e in list(best.items())[-3:]:
                if e.get("code_change_summary"):
                    lines.append(f"- **{comp}**: {e['code_change_summary']}")
        return "\n".join(lines)

    def _render_feedback_context(self) -> str:
        """For feedback eval — SOTA vs current results, no RAG or hypothesis history."""
        lines = ["## Memory: Feedback Phase Context"]
        if self._data["properties"]:
            sota = min(self._data["properties"], key=lambda p: p["overall_rmse"])
            sc_overall = sota.get("s_curve_rmse_overall", float("nan"))
            sc_mid = sota.get("s_curve_rmse_mid_belly", float("nan"))
            sc_overall_s = f"{sc_overall:.5f}" if sc_overall == sc_overall else "n/a"
            sc_mid_s = f"{sc_mid:.5f}" if sc_mid == sc_mid else "n/a"
            lines.append(
                f"\n### SOTA so far (iter {sota['iteration']})"
                f"\n- overall_rmse: {sota['overall_rmse']:.5f}"
                f"\n- s_curve_rmse_overall: {sc_overall_s}"
                f"\n- s_curve_rmse_mid_belly: {sc_mid_s}"
            )
            if sota.get("rmse_by_coupon_bucket"):
                coupon_str = ", ".join(
                    f"{k}:{v:.5f}" for k, v in sota["rmse_by_coupon_bucket"].items()
                )
                lines.append(f"- per-coupon RMSE: {coupon_str}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Code-diff summarization helper
# ---------------------------------------------------------------------------


def summarize_code_change(hypothesis: str, diff_text: str, max_lines: int = 3) -> str:
    """Produce a mathematical-change summary from a code diff.

    This is a lightweight heuristic — in production you would call an LLM
    with a dedicated summarization prompt ("describe only the mathematical
    changes, not the formatting or variable renames"). For now it extracts
    the first N meaningful added lines that contain equations or function
    calls, which is enough to prevent raw diffs from bloating the trace.
    """
    if not diff_text:
        return hypothesis[:200]
    interesting: list[str] = []
    for line in diff_text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("+"):
            continue
        body = stripped[1:].strip()
        if not body or body.startswith("#"):
            continue
        # Prefer lines that look like math or function definitions
        if any(tok in body for tok in ("=", "def ", "return ", "np.", "math.", "torch.")):
            interesting.append(body[:140])
        if len(interesting) >= max_lines:
            break
    if not interesting:
        return hypothesis[:200]
    return " | ".join(interesting)
