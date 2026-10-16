"""Fusion configuration: the unit the agent proposes and every results-table row is built from.

A ``FusionSpec`` is immutable, hashable, JSON round-trippable and validated against the
search space, so a proposal from an LLM can be checked before anything is trained.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any

from pathfusion.utils.io import stable_hash

FAMILIES = ("meta_only", "single", "early_concat", "late", "intermediate")
NORMS = ("none", "l2", "zscore")
PCA_DIMS = (None, 64, 128, 256)
LATE_RULES = ("prob_mean", "logit_mean", "weighted", "stacking")
STACK_CS = (0.1, 1.0, 10.0)
STREAM_FUSIONS = ("gated", "attention", "concat")
MODALITY_DROPOUTS = (0.0, 0.1, 0.25)


@dataclass(frozen=True)
class FusionSpec:
    family: str
    models: tuple[str, ...] = field(default_factory=tuple)
    use_meta: bool = True
    norm: str = "none"              # per-model tile-feature normalisation before fusion
    pca_dim: int | None = None      # per-model PCA (fit on training-fold tiles)
    late_rule: str = "prob_mean"    # late fusion only
    stack_c: float = 1.0            # late/stacking only: inverse L2 strength
    stream_fusion: str = "gated"    # intermediate only
    modality_dropout: float = 0.0   # intermediate only

    def __post_init__(self) -> None:
        object.__setattr__(self, "models", tuple(sorted(self.models)))
        self.validate()

    # ------------------------------------------------------------------ validation
    def validate(self, available_models: tuple[str, ...] | None = None) -> None:
        f = self.family
        if f not in FAMILIES:
            raise ValueError(f"family must be one of {FAMILIES}, got {f!r}")
        if f == "meta_only":
            if self.models or not self.use_meta:
                raise ValueError("meta_only uses no image models and must use metadata")
        elif f == "single":
            if len(self.models) != 1:
                raise ValueError("single needs exactly one model")
        elif len(self.models) < 2:
            raise ValueError(f"{f} fuses >= 2 models")
        if len(set(self.models)) != len(self.models):
            raise ValueError("duplicate models")
        if available_models is not None and set(self.models) - set(available_models):
            raise ValueError(f"unknown models {set(self.models) - set(available_models)}")
        if self.norm not in NORMS:
            raise ValueError(f"norm must be one of {NORMS}")
        if self.pca_dim not in PCA_DIMS:
            raise ValueError(f"pca_dim must be one of {PCA_DIMS}")
        if self.late_rule not in LATE_RULES:
            raise ValueError(f"late_rule must be one of {LATE_RULES}")
        if self.stack_c not in STACK_CS:
            raise ValueError(f"stack_c must be one of {STACK_CS}")
        if self.stream_fusion not in STREAM_FUSIONS:
            raise ValueError(f"stream_fusion must be one of {STREAM_FUSIONS}")
        if self.modality_dropout not in MODALITY_DROPOUTS:
            raise ValueError(f"modality_dropout must be one of {MODALITY_DROPOUTS}")

    # ------------------------------------------------------------------ canonical form
    def canonical(self) -> FusionSpec:
        """Reset knobs that do not apply to the family, so equivalent specs compare equal."""
        d = FusionSpec(family=self.family, models=self.models, use_meta=self.use_meta)
        if self.family == "meta_only":
            return d
        d = replace(d, norm=self.norm, pca_dim=self.pca_dim)
        if self.family == "late":
            d = replace(d, late_rule=self.late_rule,
                        stack_c=self.stack_c if self.late_rule == "stacking" else 1.0)
        if self.family == "intermediate":
            d = replace(d, stream_fusion=self.stream_fusion, modality_dropout=self.modality_dropout)
        return d

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self.canonical())
        d["models"] = list(d["models"])
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FusionSpec:
        d = dict(d)
        d["models"] = tuple(d.get("models", ()))
        if d.get("pca_dim") in ("null", "None", 0):
            d["pca_dim"] = None
        known = set(cls.__dataclass_fields__)
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown spec fields {sorted(unknown)}")
        return cls(**d).canonical()

    @property
    def key(self) -> str:
        return stable_hash(self.to_dict())

    def complexity(self) -> float:
        """Rough ordinal model complexity, used by the one-standard-error selection rule."""
        base = {"meta_only": 0, "single": 1, "early_concat": 3, "late": 2, "intermediate": 5}[self.family]
        if self.family == "late":
            base += {"prob_mean": 0, "logit_mean": 0, "weighted": 1, "stacking": 2}[self.late_rule]
        if self.family == "intermediate":
            base += {"concat": 0, "gated": 1, "attention": 2}[self.stream_fusion]
        if self.pca_dim is not None:
            base += 0.5
        if self.norm != "none":
            base += 0.25
        return base + 0.1 * len(self.models)

    def short(self) -> str:
        """Human-readable one-liner for logs and tables."""
        if self.family == "meta_only":
            return "metadata only (age + sex)"
        parts = [self.family, "+".join(self.models)]
        if self.norm != "none":
            parts.append(f"norm={self.norm}")
        if self.pca_dim:
            parts.append(f"pca={self.pca_dim}")
        if self.family == "late":
            parts.append(self.late_rule + (f"(C={self.stack_c})" if self.late_rule == "stacking" else ""))
        if self.family == "intermediate":
            parts.append(self.stream_fusion)
            if self.modality_dropout:
                parts.append(f"moddrop={self.modality_dropout}")
        parts.append("+meta" if self.use_meta else "no-meta")
        return " ".join(parts)
