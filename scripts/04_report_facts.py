"""Stage 4: collect every number the technical report needs into one facts sheet.

This is a *fact sheet* (numbers, settings, file pointers), not report prose: the brief asks
that the report itself be written by the author.

Usage:
    python scripts/04_report_facts.py [--embeddings-dir DIR]
Output: artifacts/results/report_facts.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from pathfusion.config import load_config  # noqa: E402
from pathfusion.utils.io import load_json  # noqa: E402


def section(title: str, body: str) -> str:
    return f"## {title}\n\n{body.strip()}\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embeddings-dir", type=Path, default=None)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(overrides=args.overrides)
    res_dir = cfg.paths.resolve("results_dir")
    out: list[str] = ["# Report fact sheet (auto-generated; numbers only)\n"]

    cohort = pd.read_csv(cfg.paths.resolve("cohort_csv"), dtype={"patient_id": str})
    split = pd.read_csv(cfg.paths.resolve("split_csv"), dtype={"patient_id": str})
    out.append(section("Data", f"""
- Cohort: {len(cohort)} patients, one slide each (lowest available WSI_ID). See artifacts/data/data_audit.md.
- Split (patients): {split['split'].value_counts().to_dict()}; stratified per class, seed {cfg.split.seed}.
- Development CV: {cfg.split.cv_folds} folds x {cfg.split.cv_repeats} repeat(s) over train+val.
- Policies: duplicate_id={cfg.data.duplicate_id_policy}, conflicts={cfg.data.conflict_policy}.
"""))
    t = cfg.tiling
    out.append(section("Tiling", f"""
- {t.native_magnification}x -> {t.target_magnification}x by JPEG DCT-domain shrink-on-load ({t.downsample}x linear).
- Tissue mask at {t.target_magnification / t.thumbnail_downsample:.2f}x: HSV saturation, median filter, Otsu
  (floor {t.saturation_floor}), pen-hue removal, closing/opening, small-object removal.
- Tiles {t.tile_size}px at 20x, stride {t.stride}, kept if tissue >= {t.min_tissue_fraction:.0%};
  cap {t.max_tiles}/slide.
"""))
    emb = args.embeddings_dir or cfg.paths.resolve("embeddings_dir")
    for name in ("extraction_summary.json", "encoders.json"):
        p = Path(emb) / name
        if p.exists():
            out.append(section(name, "```json\n" + p.read_text().strip() + "\n```"))
    tr = cfg.train
    out.append(section("Shared MIL recipe (identical for all rows)", f"""
- Gated ABMIL: Linear(D->{tr.hidden_dim})+ReLU+Dropout({tr.dropout}), attention dim {tr.attn_dim}; metadata
  MLP 2->{tr.meta_dim}. AdamW lr {tr.lr}, wd {tr.weight_decay}, {tr.epochs} epochs, batch {tr.batch_size} bags,
  {tr.train_tiles} sampled tiles/bag in training, <= {tr.eval_tiles} at inference, class-weighted CE,
  {tr.warmup_epochs}-epoch warm-up + cosine. Seeds {cfg.eval.seeds}.
"""))
    agent_root = cfg.paths.resolve("agent_dir")
    if (agent_root / "LATEST").exists():
        run = (agent_root / "LATEST").read_text().strip()
        sel = load_json(agent_root / run / "selected.json")
        trials = pd.read_csv(agent_root / run / "trials.csv")
        has_tx = (agent_root / run / "llm_transcript.jsonl").exists()
        out.append(section("Agent", f"""
- Run: artifacts/agent/{run} (decisions.md, decisions.jsonl, trials.csv{', llm_transcript.jsonl' if has_tx else ''}).
- Proposer: {sel.get('proposer')}; budget {sel['budget']}, used {sel['fusion_trials']}; stop: {sel['stop_reason']}.
- Trials by source: {trials['source'].value_counts().to_dict()}; accepted: {int(trials['accepted'].sum())}.
- Best single + meta (CV): {sel['best_reference_short']} = {sel['best_reference_cv_mean']:.3f}.
- Selected: {sel['selected_short']} = {sel['selected_cv_mean']:.3f} +/- {sel['selected_cv_se']:.3f};
  gain {sel['cv_gain_vs_best_single']:+.3f} +/- {sel['cv_gain_se']:.3f}.
- Selection optimism: {json.dumps(sel['selection_optimism'])}
"""))
    if (res_dir / "results.md").exists():
        out.append((res_dir / "results.md").read_text().replace("# Results", "## Results"))
        errs = pd.read_csv(res_dir / "error_analysis.csv")
        wrong = errs[~errs["correct"]]
        pairs = wrong.groupby(["true", "pred"]).size().sort_values(ascending=False).head(8)
        out.append(section("Error analysis (fused, test)", f"""
- Errors: {len(wrong)}/{len(errs)}; of which predicted pattern is listed in the slide's Multi_Subtypes:
  {int(wrong['pred_in_multi_subtypes'].sum())}.
- Most frequent confusions (true -> pred): {', '.join(f'{a}->{b} ({n})' for (a, b), n in pairs.items())}.
- Errors by diagnosis: {wrong.groupby('diagnosis').size().to_dict()}
  (test patients: {errs.groupby('diagnosis').size().to_dict()}).
- Figures: artifacts/results/figures/ (confusion, forest plots, per-class AUROC, curves, agent trajectory).
"""))
    study = cfg.paths.resolve("artifacts") / "extra_slides_study" / "results.md"
    if study.exists():  # post-hoc, development CV only (scripts/09_extra_slides_study.py)
        out.append(study.read_text().replace("# Extra-slides study", "## Post-hoc extra-slides study"))
    (res_dir / "report_facts.md").write_text("\n".join(out))
    print(res_dir / "report_facts.md")


if __name__ == "__main__":
    main()
