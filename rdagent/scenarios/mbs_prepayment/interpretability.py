"""MBS Interpretability Loop — Priority 4: IG attribution as a first-class signal.

This module implements Direction #8 (Interpretability as a First-Class Loop Signal).
It makes feature attribution a DRIVER of the R&D loop, not just a post-hoc analysis.

Three responsibilities:

    1. Compute: After each successful experiment, compute feature attributions
       for the trained model (Integrated Gradients if torch available, else
       permutation importance fallback).

    2. Sanity-check: Each attribution is checked against known economic priors
       for MBS prepayment (e.g., higher unemployment should NOT increase
       predicted SMM_DECIMAL). Violations are flagged as modeling defects.

    3. Feed back: The attribution summary and any flagged violations are
       persisted to an attribution-memory file that the next iteration's
       hypothesis_gen prompt reads as structured input.

This makes the loop self-auditing: an experiment is not "good" just because
overall_rmse dropped — it must also show economically reasonable attributions.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Economic priors for MBS prepayment — sanity-check targets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AttributionPrior:
    """Expected sign (and optional magnitude ratio) for a feature's attribution."""

    feature: str
    expected_sign: int  # +1, -1, or 0 (no prior)
    explanation: str


#: Economic priors for canonical MBS features. The prior says "if the sign of
#: the mean attribution is opposite, the model has an economic bug."
MBS_ECONOMIC_PRIORS: tuple[AttributionPrior, ...] = (
    AttributionPrior(
        feature="rate_incentive",
        expected_sign=+1,
        explanation="Higher rate incentive → more refi → higher SMM_DECIMAL",
    ),
    AttributionPrior(
        feature="burnout_index",
        expected_sign=-1,
        explanation="Higher burnout → less remaining refi capacity → lower SMM_DECIMAL",
    ),
    AttributionPrior(
        feature="unemployment",
        expected_sign=-1,
        explanation="Higher unemployment → reduced mobility/refi ability → lower SMM_DECIMAL",
    ),
    AttributionPrior(
        feature="unemployment_rate",
        expected_sign=-1,
        explanation="Higher unemployment → reduced mobility/refi ability → lower SMM_DECIMAL",
    ),
    AttributionPrior(
        feature="unemployment_lag1m",
        expected_sign=-1,
        explanation="Higher unemployment → reduced mobility/refi ability → lower SMM_DECIMAL",
    ),
    AttributionPrior(
        feature="wala",
        expected_sign=+1,
        explanation="Older loans prepay more (seasoning effect)",
    ),
    AttributionPrior(
        feature="seasoning_ramp",
        expected_sign=+1,
        explanation="Fresh loans ramp up in SMM_DECIMAL over first 30 months",
    ),
    AttributionPrior(
        feature="yield_curve_slope",
        expected_sign=0,
        explanation="No strong prior; depends on segment",
    ),
    AttributionPrior(
        feature="hpi",
        expected_sign=+1,
        explanation="Rising HPI → cash-out refi incentive → higher SMM_DECIMAL",
    ),
    AttributionPrior(
        feature="hpi_lag1m",
        expected_sign=+1,
        explanation="Rising HPI → cash-out refi incentive → higher SMM_DECIMAL",
    ),
)


def _priors_by_feature() -> dict[str, AttributionPrior]:
    return {p.feature: p for p in MBS_ECONOMIC_PRIORS}


# ---------------------------------------------------------------------------
# Attribution computation
# ---------------------------------------------------------------------------


@dataclass
class AttributionResult:
    feature_names: list[str]
    mean_attribution: np.ndarray       # mean |attribution| per feature
    signed_mean_attribution: np.ndarray  # signed mean (sign matters for priors)
    method: str

    def top_n(self, n: int = 10) -> list[tuple[str, float]]:
        order = np.argsort(-np.abs(self.mean_attribution))
        return [(self.feature_names[i], float(self.mean_attribution[i])) for i in order[:n]]


def compute_attributions(
    model: Any,
    X: np.ndarray,
    feature_names: list[str],
    *,
    baseline: np.ndarray | None = None,
    n_samples: int = 500,
) -> AttributionResult:
    """Compute feature attributions for a fitted model.

    Tries Integrated Gradients (torch) first. Falls back to permutation
    importance using sklearn-compatible predict().
    """
    X = np.asarray(X, dtype=float)
    if len(X) > n_samples:
        idx = np.random.default_rng(0).choice(len(X), size=n_samples, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    # Try Integrated Gradients via captum (model must be a torch.nn.Module)
    try:
        import torch  # type: ignore
        from captum.attr import IntegratedGradients  # type: ignore
        if hasattr(model, "forward") and isinstance(model, torch.nn.Module):
            model.eval()
            x_t = torch.tensor(X_sample, dtype=torch.float32, requires_grad=True)
            if baseline is None:
                base_t = torch.zeros_like(x_t)
            else:
                base_t = torch.tensor(baseline, dtype=torch.float32).expand_as(x_t)
            ig = IntegratedGradients(model)
            attributions, _ = ig.attribute(x_t, baselines=base_t, return_convergence_delta=True)
            attr_np = attributions.detach().cpu().numpy()
            return AttributionResult(
                feature_names=feature_names,
                mean_attribution=np.abs(attr_np).mean(axis=0),
                signed_mean_attribution=attr_np.mean(axis=0),
                method="integrated_gradients",
            )
    except Exception:
        pass

    # Fallback: permutation importance on predict()
    return _permutation_importance(model, X_sample, feature_names)


def _permutation_importance(
    model: Any, X: np.ndarray, feature_names: list[str]
) -> AttributionResult:
    """Permutation importance with correlation-based sign.

    Magnitude: mean |prediction delta| when feature j is shuffled.
    Sign: Pearson correlation between feature j and baseline predictions —
    captures the direction of the feature's effect on predictions. Permuting
    alone cannot recover direction because it preserves the feature's mean.
    """
    baseline_pred = np.asarray(model.predict(X), dtype=float)
    magnitude = np.zeros(X.shape[1])
    signed = np.zeros(X.shape[1])
    rng = np.random.default_rng(1)
    y_centered = baseline_pred - baseline_pred.mean()
    y_std = baseline_pred.std() + 1e-12
    for j in range(X.shape[1]):
        # Magnitude via permutation
        X_perm = X.copy()
        rng.shuffle(X_perm[:, j])
        perm_pred = np.asarray(model.predict(X_perm), dtype=float)
        magnitude[j] = float(np.mean(np.abs(perm_pred - baseline_pred)))
        # Sign via correlation of feature with prediction
        x_j = X[:, j]
        x_centered = x_j - x_j.mean()
        x_std = x_j.std() + 1e-12
        corr = float(np.mean(x_centered * y_centered) / (x_std * y_std))
        signed[j] = corr * magnitude[j]
    return AttributionResult(
        feature_names=feature_names,
        mean_attribution=magnitude,
        signed_mean_attribution=signed,
        method="permutation_importance",
    )


# ---------------------------------------------------------------------------
# Sanity check against economic priors
# ---------------------------------------------------------------------------


@dataclass
class AttributionSanityReport:
    violations: list[dict[str, Any]] = field(default_factory=list)
    passed: list[dict[str, Any]] = field(default_factory=list)
    unknown: list[dict[str, Any]] = field(default_factory=list)

    @property
    def has_violations(self) -> bool:
        return len(self.violations) > 0


def sanity_check_attributions(result: AttributionResult) -> AttributionSanityReport:
    """Check each feature's signed mean attribution against its economic prior."""
    priors = _priors_by_feature()
    report = AttributionSanityReport()
    threshold = float(np.max(np.abs(result.signed_mean_attribution)) * 0.02 + 1e-12)

    for i, name in enumerate(result.feature_names):
        signed = float(result.signed_mean_attribution[i])
        magnitude = float(result.mean_attribution[i])
        entry = {
            "feature": name,
            "signed_mean": signed,
            "magnitude": magnitude,
        }
        if name not in priors:
            report.unknown.append(entry)
            continue
        prior = priors[name]
        if prior.expected_sign == 0:
            report.unknown.append({**entry, "prior": prior.explanation})
            continue
        if abs(signed) < threshold:
            report.unknown.append({
                **entry,
                "prior": prior.explanation,
                "note": "attribution magnitude too small to judge sign",
            })
            continue
        actual_sign = 1 if signed > 0 else -1
        entry["expected_sign"] = prior.expected_sign
        entry["actual_sign"] = actual_sign
        entry["prior_explanation"] = prior.explanation
        if actual_sign == prior.expected_sign:
            report.passed.append(entry)
        else:
            report.violations.append({
                **entry,
                "severity": "high",
                "message": (
                    f"Economic prior violated for '{name}': expected sign "
                    f"{prior.expected_sign:+d}, got {actual_sign:+d}. "
                    f"{prior.explanation}"
                ),
            })
    return report


# ---------------------------------------------------------------------------
# Persistence and feedback integration
# ---------------------------------------------------------------------------


@dataclass
class AttributionMemory:
    """Per-iteration attribution history persisted to disk for next-iteration use."""

    memory_path: Path

    def __post_init__(self):
        self.memory_path = Path(self.memory_path)
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)

    def append(
        self,
        iteration: int,
        result: AttributionResult,
        report: AttributionSanityReport,
        top_n: int = 10,
    ) -> None:
        entry = {
            "iteration": iteration,
            "method": result.method,
            "top_features": [
                {"feature": f, "magnitude": m} for f, m in result.top_n(top_n)
            ],
            "violations": report.violations,
            "has_violations": report.has_violations,
        }
        history = self.load()
        history.append(entry)
        self.memory_path.write_text(json.dumps(history, indent=2, default=str))

    def load(self) -> list[dict[str, Any]]:
        if not self.memory_path.exists():
            return []
        try:
            return json.loads(self.memory_path.read_text())
        except json.JSONDecodeError:
            return []

    def format_for_prompt(self, last_n: int = 3) -> str:
        """Render attribution history as a Markdown block for hypothesis_gen."""
        history = self.load()
        if not history:
            return "(no prior attribution history)"
        lines: list[str] = ["## Feature Attribution History (most recent first)"]
        for entry in reversed(history[-last_n:]):
            lines.append(f"\n### Iteration {entry['iteration']} (method={entry['method']})")
            lines.append("**Top features by |attribution|:**")
            for f in entry["top_features"]:
                lines.append(f"  - {f['feature']}: {f['magnitude']:.6f}")
            if entry["has_violations"]:
                lines.append("\n**⚠ Economic prior violations:**")
                for v in entry["violations"]:
                    lines.append(f"  - {v['message']}")
            else:
                lines.append("\nAll attributions consistent with economic priors.")
        return "\n".join(lines)


def run_interpretability_pass(
    model: Any,
    X: np.ndarray,
    feature_names: list[str],
    iteration: int,
    memory_path: str | Path,
) -> dict[str, Any]:
    """Full interpretability pass: compute → sanity-check → persist → return.

    Call this from the MBS workflow's post-evaluation hook, after the
    MBSEvaluationHarness has produced the scorecard.
    """
    result = compute_attributions(model, X, feature_names)
    report = sanity_check_attributions(result)
    memory = AttributionMemory(Path(memory_path))
    memory.append(iteration, result, report)
    return {
        "top_attributions": result.top_n(10),
        "has_violations": report.has_violations,
        "violation_count": len(report.violations),
        "violations": report.violations,
        "method": result.method,
    }
