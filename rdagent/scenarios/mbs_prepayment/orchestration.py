"""MBS Orchestration — Priority 8.

Implements Direction #3 (Orchestration Design). Replaces the generic
Draft/Improvement/Ensemble stage selection with:

    1. **Multi-phase iteration strategy with hard gates** (3B):
         Phase 1 Baseline → Phase 2 Rate Response → Phase 3 Dynamics
         → Phase 4 Macro & Regime → Phase 5 Ensemble & Robustness
       Each phase declares a goal, a set of allowed components (narrower
       than the full curriculum from Priority 6), and gate criteria that
       must pass on the current SOTA scorecard before the loop is allowed
       to advance to the next phase.

    2. **Deterministic domain validation node** (3C):
       `DomainValidator` runs between Execute and Feedback and performs
       Python-only sanity checks (no LLM call): predictions in [0, 1],
       monotonic rate sensitivity, no NaN/Inf, all CUSIPs present,
       training time under budget. If any check fails, the experiment is
       auto-rejected with a specific diagnostic message that can be fed
       straight back to the coder, bypassing the feedback LLM and saving
       a round trip.

    3. **Human-in-the-loop review at phase gates** (3E):
       At the end of each phase, `build_phase_review_payload()` assembles
       an `interrupt()`-shaped payload (current_phase, next_phase,
       gate_results, model_properties, recommendation) so a human quant
       modeler can approve / reject-with-guidance / override. The
       interrupt mechanism itself is not wired here — this module only
       builds the payload; the surrounding loop (LangGraph or custom) is
       responsible for pausing.

This module deliberately depends on the Priority 1 scorecard shape, the
Priority 6 `MBSSearchState` (for curriculum constraints), and the
Priority 7 `MBSMemory` (for SOTA properties to put in the review payload).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Sequence

from rdagent.scenarios.mbs_prepayment.memory import MBSMemory, ModelProperties
from rdagent.scenarios.mbs_prepayment.search_strategy import (
    HypothesisFilter,
    MBSSearchState,
    decide_next_iteration,
)


# ---------------------------------------------------------------------------
# Phase definitions
# ---------------------------------------------------------------------------


class Phase(str, Enum):
    BASELINE = "baseline"
    RATE_RESPONSE = "rate_response"
    DYNAMICS = "dynamics"
    MACRO_REGIME = "macro_regime"
    ENSEMBLE = "ensemble"


#: Canonical ordering of phases. A phase can only advance to the next one
#: in this list.
PHASE_ORDER: tuple[Phase, ...] = (
    Phase.BASELINE,
    Phase.RATE_RESPONSE,
    Phase.DYNAMICS,
    Phase.MACRO_REGIME,
    Phase.ENSEMBLE,
)


@dataclass(frozen=True)
class PhaseSpec:
    """Static description of one phase of the MBS curriculum."""

    phase: Phase
    goal: str
    iteration_budget: tuple[int, int]      # (min_iterations, max_iterations) inside this phase
    allowed_components: tuple[str, ...]    # components the hypothesis_gen is allowed to touch
    gate_criteria_description: str         # human-readable for the review payload

    def __str__(self) -> str:
        return f"{self.phase.value} ({self.iteration_budget[0]}-{self.iteration_budget[1]} iters)"


#: Specification of all 5 MBS phases. Keep in sync with mbs_customization.md §3B.
PHASE_SPECS: dict[Phase, PhaseSpec] = {
    Phase.BASELINE: PhaseSpec(
        phase=Phase.BASELINE,
        goal=(
            "Establish a working pipeline with rate_incentive as the primary feature "
            "and a simple model (Ridge or small GBM). Validate the temporal train/test "
            "split and the MBSEvaluationHarness scorecard shape."
        ),
        iteration_budget=(1, 3),
        allowed_components=("DataLoader", "RateCurveFeatures", "PrepaymentModel"),
        gate_criteria_description=(
            "overall_rmse < 0.040 AND all predictions in [0, 1] AND monotonic rate sensitivity."
        ),
    ),
    Phase.RATE_RESPONSE: PhaseSpec(
        phase=Phase.RATE_RESPONSE,
        goal=(
            "Get the S-curve right — a nonlinear refinancing response to rate_incentive. "
            "The model must exhibit a sigmoidal shape with inflection in [50, 150] bps."
        ),
        iteration_budget=(4, 8),
        allowed_components=("RateCurveFeatures", "PrepaymentModel"),
        gate_criteria_description=(
            "spearman(monotonicity) >= 0.7 AND s_curve_r2 >= 0.6 AND "
            "inflection_point_bps in [50, 150]."
        ),
    ),
    Phase.DYNAMICS: PhaseSpec(
        phase=Phase.DYNAMICS,
        goal=(
            "Add burnout, seasoning, vintage, and seasonal effects. Each iteration "
            "should add exactly one dynamic feature and prove it helps per-coupon RMSE."
        ),
        iteration_budget=(9, 14),
        allowed_components=("PoolDynamics", "RateCurveFeatures", "PrepaymentModel"),
        gate_criteria_description=(
            "overall_rmse improved over RATE_RESPONSE SOTA AND worst per-coupon-bucket "
            "RMSE < 0.035 AND monotonicity preserved."
        ),
    ),
    Phase.MACRO_REGIME: PhaseSpec(
        phase=Phase.MACRO_REGIME,
        goal=(
            "Add macro environment sensitivity (unemployment, HPI, refi index) and "
            "validate on regime transitions (2013 taper, 2020 COVID, 2022 rate hiking)."
        ),
        iteration_budget=(15, 18),
        allowed_components=("MacroFeatures", "PrepaymentModel", "ScenarioValidator"),
        gate_criteria_description=(
            "regime_transition_rmse_mean <= 2 * overall_rmse."
        ),
    ),
    Phase.ENSEMBLE: PhaseSpec(
        phase=Phase.ENSEMBLE,
        goal=(
            "Ensemble multiple model architectures and stress-test CUSIP-level tails. "
            "Final phase — optimize for robustness not raw accuracy."
        ),
        iteration_budget=(19, 22),
        allowed_components=("Ensemble", "ScenarioValidator", "PrepaymentModel"),
        gate_criteria_description=(
            "Ensemble overall_rmse < best single-model overall_rmse "
            "AND cusip_differentiation_std > 0."
        ),
    ),
}


def next_phase(phase: Phase) -> Phase | None:
    """Return the phase that follows `phase`, or None if this is the last phase."""
    try:
        idx = PHASE_ORDER.index(phase)
    except ValueError:
        return None
    if idx + 1 >= len(PHASE_ORDER):
        return None
    return PHASE_ORDER[idx + 1]


# ---------------------------------------------------------------------------
# Domain validation (deterministic — no LLM)
# ---------------------------------------------------------------------------


@dataclass
class ValidationCheck:
    name: str
    passed: bool
    detail: str = ""

    def __str__(self) -> str:
        mark = "OK" if self.passed else "FAIL"
        return f"[{mark}] {self.name}: {self.detail}"


@dataclass
class ValidationReport:
    """Result of `DomainValidator.validate()`.

    If `ok` is False, `auto_reject_reason` contains a compact diagnostic
    that can be fed straight back to the coder without calling the
    feedback LLM.
    """

    checks: list[ValidationCheck]
    ok: bool
    auto_reject_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "auto_reject_reason": self.auto_reject_reason,
            "checks": [
                {"name": c.name, "passed": c.passed, "detail": c.detail} for c in self.checks
            ],
        }


@dataclass
class DomainValidator:
    """Deterministic Python checks on an experiment result.

    Runs between Execute and Feedback to auto-reject obviously-broken
    experiments without spending an LLM call. The checks are intentionally
    narrow — they look for violations that are unambiguous bugs, not
    model-quality regressions (those are the feedback LLM's job).
    """

    target_min: float = 0.0
    target_max: float = 1.0
    min_rate_sensitivity_corr: float = 0.3  # Spearman(pred, rate_incentive) must exceed this
    max_training_seconds: float = 3600.0    # 1-hour default budget

    def validate(
        self,
        *,
        y_pred: Sequence[float],
        rate_incentive: Sequence[float] | None = None,
        cusip_ids_train: Sequence[Any] | None = None,
        cusip_ids_pred: Sequence[Any] | None = None,
        training_seconds: float | None = None,
    ) -> ValidationReport:
        checks: list[ValidationCheck] = []

        # Check 1: predictions in [target_min, target_max]
        in_range = all(
            (not _is_nan(v)) and self.target_min <= v <= self.target_max for v in y_pred
        )
        checks.append(
            ValidationCheck(
                name="target_range",
                passed=in_range,
                detail=(
                    f"all {len(y_pred)} predictions in [{self.target_min}, {self.target_max}]"
                    if in_range
                    else (
                        "some predictions fall outside [0, 1] or are NaN — "
                        "did you forget to clip or use a sigmoid head?"
                    )
                ),
            )
        )

        # Check 2: no NaN/Inf anywhere in predictions
        no_nan_inf = all(_is_finite(v) for v in y_pred)
        checks.append(
            ValidationCheck(
                name="no_nan_inf",
                passed=no_nan_inf,
                detail="no NaN/Inf in predictions" if no_nan_inf else "NaN or Inf found in predictions",
            )
        )

        # Check 3: monotonic rate sensitivity
        if rate_incentive is not None and len(rate_incentive) == len(y_pred) and len(y_pred) >= 3:
            corr = _spearman(list(rate_incentive), list(y_pred))
            ok = corr >= self.min_rate_sensitivity_corr
            checks.append(
                ValidationCheck(
                    name="rate_sensitivity_monotonic",
                    passed=ok,
                    detail=(
                        f"spearman(pred, rate_incentive)={corr:+.3f} "
                        f"(threshold {self.min_rate_sensitivity_corr:+.2f})"
                    ),
                )
            )
        else:
            checks.append(
                ValidationCheck(
                    name="rate_sensitivity_monotonic",
                    passed=True,
                    detail="rate_incentive not provided — skipped",
                )
            )

        # Check 4: all CUSIPs present at prediction time
        if cusip_ids_train is not None and cusip_ids_pred is not None:
            train_set = set(cusip_ids_train)
            pred_set = set(cusip_ids_pred)
            missing = train_set - pred_set
            checks.append(
                ValidationCheck(
                    name="all_cusips_present",
                    passed=not missing,
                    detail=(
                        f"all {len(train_set)} training CUSIPs have predictions"
                        if not missing
                        else (
                            f"{len(missing)} training CUSIPs missing predictions — "
                            "silent data drop detected"
                        )
                    ),
                )
            )

        # Check 5: training time within budget
        if training_seconds is not None:
            ok = training_seconds <= self.max_training_seconds
            checks.append(
                ValidationCheck(
                    name="training_time_budget",
                    passed=ok,
                    detail=f"training took {training_seconds:.0f}s (budget {self.max_training_seconds:.0f}s)",
                )
            )

        all_ok = all(c.passed for c in checks)
        reason = (
            ""
            if all_ok
            else "Domain validation failed: " + "; ".join(c.detail for c in checks if not c.passed)
        )
        return ValidationReport(checks=checks, ok=all_ok, auto_reject_reason=reason)

    def validate_from_scorecard(self, scorecard: dict[str, Any]) -> ValidationReport:
        """Validate using precomputed scorecard fields (no raw y_pred needed).

        The DS runner does not surface raw predictions back to the loop, but
        the MBS scaffold has already written the aggregate diagnostics to
        ``scores.json``. This method inspects those diagnostics directly so
        the auto-reject gate actually fires on real failure modes instead of
        a placeholder payload.

        Fires on:
          * NaN/missing ``overall_rmse`` or RMSE outside a plausible range
          * Non-monotonic rate sensitivity (Spearman below threshold)
          * Any harness dimension that returned an ``_error`` field
        """
        checks: list[ValidationCheck] = []

        acc = scorecard.get("accuracy", {}) or {}
        rs = scorecard.get("rate_sensitivity", {}) or {}

        # Check 1: overall_rmse finite and plausible (0, 1].
        overall = acc.get("overall_rmse", float("nan"))
        rmse_ok = _is_finite(overall) and 0.0 < float(overall) <= 1.0
        checks.append(
            ValidationCheck(
                name="overall_rmse_finite",
                passed=rmse_ok,
                detail=(
                    f"overall_rmse={overall}"
                    if rmse_ok
                    else f"overall_rmse={overall} — NaN or outside (0, 1]; "
                    "check for empty test split, all-NaN predictions, or unclipped output"
                ),
            )
        )

        # Check 2: monotonic rate sensitivity (if the harness computed it).
        if "monotonicity_spearman" in rs and "_error" not in rs:
            corr = rs.get("monotonicity_spearman", 0.0)
            mono_ok = _is_finite(corr) and float(corr) >= self.min_rate_sensitivity_corr
            checks.append(
                ValidationCheck(
                    name="rate_sensitivity_monotonic",
                    passed=mono_ok,
                    detail=(
                        f"spearman(pred, rate_incentive)={float(corr):+.3f} "
                        f"(threshold {self.min_rate_sensitivity_corr:+.2f})"
                    ),
                )
            )

        # Check 3: no harness dimension errored out.
        errored_dims = [
            dim for dim, vals in scorecard.items()
            if isinstance(vals, dict) and "_error" in vals
        ]
        checks.append(
            ValidationCheck(
                name="harness_dimensions_ok",
                passed=not errored_dims,
                detail=(
                    "all scorecard dimensions produced results"
                    if not errored_dims
                    else f"dimensions with errors: {', '.join(errored_dims)}"
                ),
            )
        )

        all_ok = all(c.passed for c in checks)
        reason = (
            ""
            if all_ok
            else "Domain validation failed (scorecard-driven): "
            + "; ".join(c.detail for c in checks if not c.passed)
        )
        return ValidationReport(checks=checks, ok=all_ok, auto_reject_reason=reason)


def _is_nan(x: float) -> bool:
    try:
        return math.isnan(float(x))
    except (TypeError, ValueError):
        return True


def _is_finite(x: float) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _spearman(a: list[float], b: list[float]) -> float:
    """Spearman rank correlation; zero for constant inputs."""
    n = len(a)
    if n < 2:
        return 0.0
    ra = _rank(a)
    rb = _rank(b)
    mean_a = sum(ra) / n
    mean_b = sum(rb) / n
    num = sum((ra[i] - mean_a) * (rb[i] - mean_b) for i in range(n))
    den_a = math.sqrt(sum((ra[i] - mean_a) ** 2 for i in range(n)))
    den_b = math.sqrt(sum((rb[i] - mean_b) ** 2 for i in range(n)))
    if den_a == 0 or den_b == 0:
        return 0.0
    return num / (den_a * den_b)


def _rank(x: list[float]) -> list[float]:
    order = sorted(range(len(x)), key=lambda i: x[i])
    ranks = [0.0] * len(x)
    for r, i in enumerate(order):
        ranks[i] = float(r)
    return ranks


# ---------------------------------------------------------------------------
# Phase gate evaluation
# ---------------------------------------------------------------------------


@dataclass
class GateResult:
    phase: Phase
    passed: bool
    criteria_results: list[ValidationCheck]
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase.value,
            "passed": self.passed,
            "summary": self.summary,
            "criteria": [
                {"name": c.name, "passed": c.passed, "detail": c.detail}
                for c in self.criteria_results
            ],
        }


@dataclass
class PhaseGate:
    """Evaluate whether the current SOTA scorecard satisfies a phase's gate."""

    baseline_max_rmse: float = 0.040
    rate_response_min_monotonicity: float = 0.7
    rate_response_min_s_curve_r2: float = 0.6
    rate_response_inflection_range_bps: tuple[float, float] = (50.0, 150.0)
    dynamics_max_worst_coupon_rmse: float = 0.035
    macro_regime_transition_ratio: float = 2.0  # regime_rmse <= ratio * overall_rmse

    def evaluate(self, phase: Phase, sota: ModelProperties | dict[str, Any] | None) -> GateResult:
        if sota is None:
            return GateResult(
                phase=phase,
                passed=False,
                criteria_results=[ValidationCheck("has_sota", False, "no successful experiment yet")],
                summary="No SOTA available — gate cannot be evaluated.",
            )

        # Accept either a ModelProperties or its dict form.
        props = sota if isinstance(sota, dict) else _props_to_dict(sota)
        checks: list[ValidationCheck] = []

        if phase == Phase.BASELINE:
            ok = props["overall_rmse"] < self.baseline_max_rmse
            checks.append(
                ValidationCheck(
                    "overall_rmse_under_baseline_cap",
                    ok,
                    f"overall_rmse={props['overall_rmse']:.5f} (cap {self.baseline_max_rmse:.5f})",
                )
            )
            mono_ok = props.get("rate_sensitivity_spearman", 0.0) > 0
            checks.append(
                ValidationCheck(
                    "monotonic_rate_sensitivity",
                    mono_ok,
                    f"spearman={props.get('rate_sensitivity_spearman', 0.0):+.2f}",
                )
            )

        elif phase == Phase.RATE_RESPONSE:
            mono = props.get("rate_sensitivity_spearman", 0.0)
            checks.append(
                ValidationCheck(
                    "monotonicity_ge_threshold",
                    mono >= self.rate_response_min_monotonicity,
                    f"spearman={mono:+.2f} (threshold {self.rate_response_min_monotonicity:+.2f})",
                )
            )
            r2 = props.get("s_curve_r2", 0.0)
            checks.append(
                ValidationCheck(
                    "s_curve_r2_ge_threshold",
                    r2 >= self.rate_response_min_s_curve_r2,
                    f"s_curve_r2={r2:.2f} (threshold {self.rate_response_min_s_curve_r2:.2f})",
                )
            )
            infl = props.get("inflection_point_bps", 0.0)
            lo, hi = self.rate_response_inflection_range_bps
            checks.append(
                ValidationCheck(
                    "inflection_in_range",
                    lo <= infl <= hi,
                    f"inflection={infl:.0f}bps (target [{lo:.0f}, {hi:.0f}]bps)",
                )
            )

        elif phase == Phase.DYNAMICS:
            coupon_rmses: dict[str, float] = props.get("rmse_by_coupon_bucket", {}) or {}
            if coupon_rmses:
                worst = max(coupon_rmses.values())
                ok = worst < self.dynamics_max_worst_coupon_rmse
                checks.append(
                    ValidationCheck(
                        "worst_coupon_bucket_rmse",
                        ok,
                        f"worst={worst:.5f} (cap {self.dynamics_max_worst_coupon_rmse:.5f})",
                    )
                )
            else:
                checks.append(
                    ValidationCheck(
                        "worst_coupon_bucket_rmse",
                        False,
                        "no per-coupon RMSE in scorecard",
                    )
                )
            mono = props.get("rate_sensitivity_spearman", 0.0)
            checks.append(
                ValidationCheck(
                    "monotonicity_preserved",
                    mono >= self.rate_response_min_monotonicity,
                    f"spearman={mono:+.2f}",
                )
            )

        elif phase == Phase.MACRO_REGIME:
            overall = props.get("overall_rmse", 0.0)
            regime = props.get("regime_transition_rmse_mean", 0.0)
            if overall > 0:
                ratio = regime / overall
                ok = ratio <= self.macro_regime_transition_ratio
                checks.append(
                    ValidationCheck(
                        "regime_ratio_within_cap",
                        ok,
                        (
                            f"regime_rmse={regime:.5f}, overall_rmse={overall:.5f}, "
                            f"ratio={ratio:.2f} (cap {self.macro_regime_transition_ratio:.1f})"
                        ),
                    )
                )
            else:
                checks.append(
                    ValidationCheck("regime_ratio_within_cap", False, "overall_rmse is zero")
                )

        elif phase == Phase.ENSEMBLE:
            diff_std = props.get("cusip_differentiation_std", 0.0)
            checks.append(
                ValidationCheck(
                    "cusip_differentiation_positive",
                    diff_std > 0,
                    f"cusip_differentiation_std={diff_std:.5f}",
                )
            )

        passed = all(c.passed for c in checks) if checks else False
        summary = (
            f"Phase {phase.value} gate "
            + ("PASSED" if passed else "NOT PASSED")
            + f" — {sum(c.passed for c in checks)}/{len(checks)} criteria satisfied."
        )
        return GateResult(phase=phase, passed=passed, criteria_results=checks, summary=summary)


def _props_to_dict(props: ModelProperties) -> dict[str, Any]:
    return {
        "iteration": props.iteration,
        "model_type": props.model_type,
        "component_touched": props.component_touched,
        "overall_rmse": props.overall_rmse,
        "rmse_by_coupon_bucket": dict(props.rmse_by_coupon_bucket),
        "rate_sensitivity_spearman": props.rate_sensitivity_spearman,
        "s_curve_r2": props.s_curve_r2,
        "inflection_point_bps": props.inflection_point_bps,
        "regime_transition_rmse_mean": props.regime_transition_rmse_mean,
        "cusip_differentiation_std": props.cusip_differentiation_std,
    }


# ---------------------------------------------------------------------------
# Orchestrator: glues phase state, search strategy, memory, and gate checks
# ---------------------------------------------------------------------------


@dataclass
class HumanReviewPayload:
    """Shape of the `interrupt()` payload at a phase-gate transition.

    The surrounding loop (LangGraph or similar) is responsible for pausing
    and presenting this to the human. This module only assembles it.
    """

    current_phase: Phase
    proposed_next_phase: Phase | None
    gate_result: GateResult
    model_properties: dict[str, Any] | None
    recommendation: str
    allowed_components_next_phase: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_phase": self.current_phase.value,
            "proposed_next_phase": self.proposed_next_phase.value if self.proposed_next_phase else None,
            "gate_result": self.gate_result.to_dict(),
            "model_properties": self.model_properties,
            "recommendation": self.recommendation,
            "allowed_components_next_phase": list(self.allowed_components_next_phase),
        }


@dataclass
class MBSOrchestrator:
    """Top-level coordinator for the MBS phased loop.

    Usage sketch:

        orch = MBSOrchestrator(memory=mbs_memory, search_state=search_state)

        # Before each iteration:
        constraint = orch.iteration_constraints()
        # → HypothesisFilter intersected with the current phase's allowed components

        # After each iteration:
        report = orch.validate_result(...)   # deterministic check
        if not report.ok:
            # auto-reject, feed reason to coder, skip feedback LLM
            ...

        # At the end of the phase's iteration budget:
        payload = orch.build_phase_review_payload()
        # → pass to interrupt() / human CLI
    """

    memory: MBSMemory
    search_state: MBSSearchState
    validator: DomainValidator = field(default_factory=DomainValidator)
    gate: PhaseGate = field(default_factory=PhaseGate)
    current_phase: Phase = Phase.BASELINE

    def phase_spec(self) -> PhaseSpec:
        return PHASE_SPECS[self.current_phase]

    # --- Per-iteration constraints --------------------------------------

    def iteration_constraints(self) -> HypothesisFilter:
        """Intersect the curriculum filter with the current phase's allowlist."""
        filt = decide_next_iteration(self.search_state)
        spec = self.phase_spec()
        phase_allowed = set(spec.allowed_components)

        new_allowed = [c for c in filt.allowed_components if c in phase_allowed]
        new_blocked = dict(filt.blocked_components)
        for c in filt.allowed_components:
            if c not in phase_allowed:
                new_blocked[c] = f"Not in active phase '{spec.phase.value}' allowlist"

        filt.allowed_components = new_allowed
        filt.blocked_components = new_blocked
        filt.guidance = (
            f"[Phase: {spec.phase.value}] Goal: {spec.goal}\n"
            f"Gate criteria: {spec.gate_criteria_description}\n"
            f"Curriculum guidance: {filt.guidance}"
        )
        return filt

    # --- Deterministic post-execution validation ------------------------

    def validate_result(self, **kwargs: Any) -> ValidationReport:
        return self.validator.validate(**kwargs)

    # --- Gate check + phase advance -------------------------------------

    def _current_sota(self) -> dict[str, Any] | None:
        best = self.memory.best_per_component()
        if not best:
            return None
        # Pick the overall best (lowest overall_rmse across components)
        overall = min(
            (e["properties"] for e in best.values() if e.get("properties")),
            key=lambda p: p.get("overall_rmse", float("inf")),
            default=None,
        )
        return overall

    def evaluate_gate(self) -> GateResult:
        return self.gate.evaluate(self.current_phase, self._current_sota())

    def build_phase_review_payload(self) -> HumanReviewPayload:
        gate_result = self.evaluate_gate()
        proposed = next_phase(self.current_phase) if gate_result.passed else self.current_phase
        if gate_result.passed and proposed is not None:
            spec_next = PHASE_SPECS[proposed]
            recommendation = (
                f"Phase {self.current_phase.value} gate PASSED. "
                f"Recommend advancing to {proposed.value}: {spec_next.goal}"
            )
            allowed_next = spec_next.allowed_components
        elif gate_result.passed and proposed is None:
            recommendation = (
                f"Phase {self.current_phase.value} gate PASSED and this is the final phase. "
                "Recommend freezing the model and producing the final report."
            )
            allowed_next = ()
        else:
            recommendation = (
                f"Phase {self.current_phase.value} gate NOT PASSED. Recommend staying in the "
                f"current phase and addressing: "
                + "; ".join(c.detail for c in gate_result.criteria_results if not c.passed)
            )
            allowed_next = self.phase_spec().allowed_components

        return HumanReviewPayload(
            current_phase=self.current_phase,
            proposed_next_phase=proposed if gate_result.passed else None,
            gate_result=gate_result,
            model_properties=self._current_sota(),
            recommendation=recommendation,
            allowed_components_next_phase=tuple(allowed_next),
        )

    def advance_phase(self) -> Phase | None:
        """Advance to the next phase if the current gate passes.

        Returns the new phase, or None if already in the last phase (or
        the gate did not pass). Call `evaluate_gate()` first if you want
        to inspect the reason for a non-advance.
        """
        gate_result = self.evaluate_gate()
        if not gate_result.passed:
            return None
        nxt = next_phase(self.current_phase)
        if nxt is None:
            return None
        self.current_phase = nxt
        return nxt

    def override_phase(self, phase: Phase) -> None:
        """Human override — jump directly to an arbitrary phase."""
        self.current_phase = phase
