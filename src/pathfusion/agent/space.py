"""The fusion search space the agent explores.

Fixed for every trial (identical budget / no hidden tuning): the MIL recipe (``TrainCfg``)
and ``use_meta=True`` (the target row is "fused + metadata"). The agent searches only
*how* the foundation-model streams are combined.

Coverage set: before any adaptive step, one representative per region of the space is
evaluated (raw / L2 / z-score+PCA concatenation, three late rules, two learned
intermediate schemes). This guarantees the space is actually spanned and gives the
adaptive phase (deterministic or LLM) an informed starting map.

``mode="single"`` is the *equal-budget control*: the same agent, budget and selection rule
search single-model configurations (which model, per-model normalisation, PCA), so the
"best single model + metadata" row receives the same tuning effort as the fused row.
"""

from __future__ import annotations

from dataclasses import replace
from itertools import combinations

from pathfusion.fusion.spec import (
    LATE_RULES,
    MODALITY_DROPOUTS,
    NORMS,
    PCA_DIMS,
    STACK_CS,
    STREAM_FUSIONS,
    FusionSpec,
)

FUSION_FAMILIES = ("early_concat", "late", "intermediate")
SINGLE_FAMILIES = ("single",)


class SearchSpace:
    def __init__(self, models: list[str] | tuple[str, ...], mode: str = "fusion"):
        self.models = tuple(sorted(models))
        if mode not in ("fusion", "single"):
            raise ValueError("mode must be 'fusion' or 'single'")
        self.mode = mode
        self.families = FUSION_FAMILIES if mode == "fusion" else SINGLE_FAMILIES
        if mode == "fusion" and len(self.models) < 2:
            raise ValueError("fusion needs at least two models")

    # --------------------------------------------------------------- fixed trial sets
    def reference_specs(self) -> list[FusionSpec]:
        """Best-single-model candidates (+ metadata): the bar fusion must clear."""
        return [FusionSpec(family="single", models=(m,), use_meta=True) for m in self.models]

    def coverage_specs(self) -> list[FusionSpec]:
        allm = self.models
        if self.mode == "single":  # same normalisation/PCA probes as fusion, per model
            return [FusionSpec("single", (m,), norm=n, pca_dim=d)
                    for m in allm for n, d in (("l2", None), ("zscore", 128))]
        return [
            FusionSpec("early_concat", allm, norm="none"),
            FusionSpec("early_concat", allm, norm="l2"),
            FusionSpec("early_concat", allm, norm="zscore", pca_dim=128),
            FusionSpec("late", allm, late_rule="prob_mean"),
            FusionSpec("late", allm, late_rule="weighted"),
            FusionSpec("late", allm, late_rule="stacking", stack_c=1.0),
            FusionSpec("intermediate", allm, stream_fusion="gated"),
            FusionSpec("intermediate", allm, stream_fusion="attention"),
        ]

    # --------------------------------------------------------------- validation
    def validate(self, spec: FusionSpec) -> FusionSpec:
        spec.validate(available_models=self.models)
        if spec.family not in self.families:
            raise ValueError(f"{self.mode} search proposals must be one of {self.families}, got {spec.family!r}")
        if not spec.use_meta:
            raise ValueError("use_meta is fixed to true for agent trials")
        return spec.canonical()

    # --------------------------------------------------------------- local moves
    def model_subsets(self) -> list[tuple[str, ...]]:
        if self.mode == "single":
            return [(m,) for m in self.models]
        return [c for k in range(len(self.models), 1, -1) for c in combinations(self.models, k)]

    def neighbors(self, spec: FusionSpec) -> list[tuple[str, FusionSpec]]:
        """One-knob changes of ``spec``, ordered by a fixed priority, each with a label."""
        out: list[tuple[str, FusionSpec]] = []

        def add(label: str, **kw: object) -> None:
            try:
                out.append((label, self.validate(replace(spec, **kw))))  # type: ignore[arg-type]
            except ValueError:
                pass

        if spec.family == "late":
            for r in LATE_RULES:
                if r != spec.late_rule:
                    add(f"late_rule {spec.late_rule}->{r}", late_rule=r)
            if spec.late_rule == "stacking":
                for c in STACK_CS:
                    if c != spec.stack_c:
                        add(f"stack_c {spec.stack_c}->{c}", stack_c=c)
        if spec.family == "intermediate":
            for f in STREAM_FUSIONS:
                if f != spec.stream_fusion:
                    add(f"stream_fusion {spec.stream_fusion}->{f}", stream_fusion=f)
            for p in MODALITY_DROPOUTS:
                if p != spec.modality_dropout:
                    add(f"modality_dropout {spec.modality_dropout}->{p}", modality_dropout=p)
        for n in NORMS:
            if n != spec.norm:
                add(f"norm {spec.norm}->{n}", norm=n)
        for d in PCA_DIMS:
            if d != spec.pca_dim:
                add(f"pca_dim {spec.pca_dim}->{d}", pca_dim=d)
        for subset in self.model_subsets():
            if subset != spec.models:
                add(f"models {'+'.join(spec.models)}->{'+'.join(subset)}", models=subset)
        return out

    # --------------------------------------------------------------- LLM description
    def describe(self) -> str:
        if self.mode == "single":
            return (
                "Config JSON fields:\n"
                '  family: "single" (one foundation model + metadata)\n'
                f"  models: list with exactly one of {list(self.models)}\n"
                "  use_meta: true (fixed)\n"
                f"  norm: one of {list(NORMS)}\n"
                f"  pca_dim: one of {list(PCA_DIMS)}  (null = none)\n"
                "A single attention-MIL over one model's tiles; tile features normalised / PCA-projected first."
            )
        return (
            "Config JSON fields (all required except where noted):\n"
            f'  family: one of {list(FUSION_FAMILIES)}\n'
            f'  models: list of >=2 of {list(self.models)}\n'
            "  use_meta: true (fixed)\n"
            f'  norm: one of {list(NORMS)}  (per-model tile normalisation before fusion)\n'
            f'  pca_dim: one of {[d for d in PCA_DIMS]}  (per-model PCA; null = none)\n'
            f'  late_rule (family=late): one of {list(LATE_RULES)}\n'
            f'  stack_c (late_rule=stacking): one of {list(STACK_CS)}\n'
            f'  stream_fusion (family=intermediate): one of {list(STREAM_FUSIONS)}\n'
            f'  modality_dropout (family=intermediate): one of {list(MODALITY_DROPOUTS)}\n'
            "Families: early_concat = one attention-MIL on per-tile concatenated features; "
            "late = one attention-MIL per model, probabilities combined (weighted/stacking fit on "
            "inner out-of-fold predictions); intermediate = per-model attention-MIL streams fused "
            "at slide level by a learned gate, attention over stream tokens, or concatenation."
        )
