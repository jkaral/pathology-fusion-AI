# Report outline (≤ 6 pages): headings mapped to the evidence, no prose

The brief requires the report to be written by its author. This file only lists the sections
the brief asks for and where the supporting numbers and figures are generated.

| § | Section (from the brief) | Evidence in the repo |
|---|---|---|
| 1 | Pipeline design | README "Design summary"; `conf/default.yaml` |
| 2 | Data and the quirks (§2.3): duplicate ID, conflicting labels, missing slides | `artifacts/data/data_audit.md`, `docs/DATA_DECISIONS.md` |
| 3 | Tissue segmentation, tiling, tile size/stride at 20x | `report_facts.md` → Tiling; QC overlays (`artifacts/qc/`, contact sheet in `artifacts/pilot/`) |
| 4 | The three models and why; per-model preprocessing; embedding dims | `embeddings/encoders.json`, `src/pathfusion/embeddings/*.py` docstrings |
| 5 | Tile → slide aggregation (ABMIL) and justification | `src/pathfusion/models/mil.py` docstring |
| 6 | Wall-clock cost and caching | `embeddings/extraction_summary.json`, `extraction_manifest.csv` |
| 7 | Metadata signal (age, sex) | `artifacts/analysis/metadata.md` + figures |
| 8 | Agent: search space, decision policy, budget, stopping, guardrails | `artifacts/agent/<run>/decisions.md`, `trajectory.png`, `llm_transcript.jsonl`; equal-budget control in `single_search/` |
| 9 | Training and validation curves | wandb project `pathology-fusion`; `artifacts/results/figures/curves.png` |
| 10 | Results table (§5.3) with uncertainty | `artifacts/results/results.md`, `forest_*.png` |
| 11 | How much the test set can resolve | `artifacts/analysis/uncertainty.md` |
| 12 | Failure analysis of the confusion matrix | `confusion_*.png`, `error_analysis.csv` (`pred_in_multi_subtypes`), per-class AUROC |
| 13 | What you would do with 10x compute | (author) |
| A | Pilot on ungated encoders (optional appendix) | `artifacts/pilot/` |
