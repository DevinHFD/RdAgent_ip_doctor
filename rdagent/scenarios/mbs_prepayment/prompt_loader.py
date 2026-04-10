"""MBS prompt loader — Priority 3.

Loads prompts.yaml and exposes helpers that mirror the
`rdagent.oai.llm_utils.APIBackend` / Jinja template lookup pattern used by
the rest of RD-Agent. This lets the MBS scenario override prompts without
touching the shared data_science/share.yaml or prompts_v2.yaml files.

Usage:
    from rdagent.scenarios.mbs_prepayment.prompt_loader import MBSPromptLoader

    loader = MBSPromptLoader()
    role = loader.get("scen.role")
    spec = loader.get("hypothesis_specification")
    dataloader_spec = loader.get("component_spec.DataLoader")

    # Render with Jinja variables:
    desc = loader.render("scenario_description", {})
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_PROMPTS_PATH = Path(__file__).parent / "prompts.yaml"


class MBSPromptLoader:
    """Loads the MBS prompts.yaml and provides dotted-path access."""

    def __init__(self, path: Path | str = _PROMPTS_PATH):
        self._path = Path(path)
        self._data = self._load()

    @lru_cache(maxsize=None)
    def _load(self) -> dict[str, Any]:  # type: ignore[override]
        with open(self._path, encoding="utf-8") as f:
            return yaml.safe_load(f)

    def get(self, dotted_key: str) -> str:
        """Get a string by dotted-path lookup, e.g. 'component_spec.DataLoader'."""
        parts = dotted_key.split(".")
        node: Any = self._data
        for p in parts:
            if not isinstance(node, dict) or p not in node:
                raise KeyError(f"Prompt key not found: {dotted_key}")
            node = node[p]
        if not isinstance(node, str):
            raise TypeError(f"Prompt at {dotted_key} is not a string (type={type(node).__name__})")
        return node

    def render(self, dotted_key: str, variables: dict[str, Any] | None = None) -> str:
        """Render a prompt with Jinja-style variables (lazy import)."""
        template_str = self.get(dotted_key)
        if not variables:
            return template_str
        try:
            from jinja2 import Template
        except ImportError:
            # Fall back to simple str.format if jinja2 not available
            return template_str.format(**variables)
        return Template(template_str).render(**variables)

    def all_keys(self) -> list[str]:
        """Return flat list of all dotted keys available in the prompts file."""
        out: list[str] = []
        def walk(d: Any, prefix: str = "") -> None:
            if isinstance(d, dict):
                for k, v in d.items():
                    walk(v, f"{prefix}.{k}" if prefix else k)
            else:
                out.append(prefix)
        walk(self._data)
        return out
