# CLAUDE.md: project reference

Summarises the coding brief ("Build an Agent for Foundation-Model Fusion in Lung Cancer
Subtyping") and the conventions of this repository. Read it before changing anything.

## Objective (from the brief)

Predict the **histologic growth pattern** of lung adenocarcinoma (7 classes) from whole-slide
images. Combine embeddings from **three pathology foundation models** with patient **age and
sex**.

- **The fused model must outperform every single-foundation-model baseline** on the held-out
  test set, under an identical split. A fused number without single-model baselines cannot be
  scored.
- Time box: ~15-30 h of part-time work. Deliverables: a reproducible repo and a ≤ 6-page technical report (PDF).

## Dataset

- `kmmuleelab/Lung_Pathology_Image_JPG` (Hugging Face, CC-BY-4.0). The paper is *Scientific
  Data* s41597-026-06906-z.
- 408 WSIs (JPG, 80x), 210 patients (1-5 slides each), lung adenocarcinoma.
- `CLWD.csv` columns: `SampleNumber` (**grouping key**), `WSI_ID` (joins to images), `Sex`,
  `Age`, `Pathological_Diagnosis` (AIS/MIA/IA), `Tumor Subtype` (**target**).
- 7 classes (patients): In situ 46, Papillary 41, Acinar 34, Solid 29, Micropapillary 26,
  Lepidic 19, Cribriform 15. Imbalanced (3:1) and small-n: one test patient moves balanced
  accuracy by ~2 points, so **quantifying uncertainty is a core part of the task**.
- Known quirks (handling them is assessed):
  1. `SampleNumber` 8377886 appears twice (ages 68 and 69).
  2. Five patients have conflicting subtype labels across their slides.
  3. Found during the audit: 22 slides are missing from the HF release, and 6 patients have no slide.
  Our handling is in `docs/DATA_DECISIONS.md`.

## Reduced protocol (headline run)

- One slide per patient (deterministic rule: lowest available `WSI_ID`).
- Downsample 80x → 20x (4x linear) before tiling.
- Cache embeddings to disk once. Never re-run a foundation model inside a hyperparameter loop.
- Free Colab/Kaggle GPU is enough. Larger runs earn no extra credit, and the reduced
  protocol stays the headline.

## Required tasks

1. **Feature extraction with three distinct FMs** (we use UNI2-h, Virchow2, Phikon-v2; UNI2-h
   and Virchow2 are gated). Document: tissue segmentation, tiling, tile size/stride at 20x;
   **per-model preprocessing (never one shared transform)**; tile→slide aggregation (the brief
   asks for **MIL, e.g. ABMIL**) with justification; embedding dims, wall-clock cost, cache location.
2. **Agent system for fusion-strategy search.**
   - The search space must include:
     - early fusion: concat, with or without per-model L2 / dimensionality reduction
     - late fusion: probability/logit averaging, weighted vote, stacking
     - one learned intermediate scheme: gated / attention / bilinear
   - The agent proposes, evaluates, reads the result and conditions its next proposal on it,
     **logging its reasoning**.
   - It needs an explicit **trial budget and stopping criterion**, and a **decision log**
     (what was tried, the score, why the next move was chosen).
   - With an LLM: keys live in the environment only, and with no key it **degrades to a
     deterministic search**.
   - **Guardrail:** with ~21 validation patients, unguided trials overfit. Defend with
     nested/repeated CV, a budget, selection-noise estimates, or a variance-aware acceptance rule.
   - **Age and sex must enter the pipeline.**

## Evaluation protocol (non-negotiable)

- **Patient-level split by `SampleNumber`** (70/10/20). No patient may span splits. A
  slide-level split is the most common *disqualifying* error.
- **Stratify by subtype**, and **fix and commit the split** (`artifacts/splits/split_v1.csv` + sha256).
- **Touch the test set once.** All selection uses train/val only (`TEST_LOCK.json` enforces this).
- **Seed everything**, and log training/validation curves to **wandb**.
- Metrics: **macro OvR AUROC** (plus per class) and **balanced accuracy**; the test confusion
  matrix; **uncertainty** as bootstrap CIs over test patients and/or mean ± sd over ≥ 3 seeds.
- Required table rows:
  - metadata only
  - FM A alone
  - FM B alone
  - FM C alone
  - best single + metadata
  - **fused (agent-selected) + metadata**

## Deliverables

- Repository: runnable end to end, pinned deps, an entry point per stage, config-driven, typed
  and documented modules, no secrets.
- Technical report (≤ 6 pages). **The brief asks that the report NOT be written by AI.** Never
  draft its prose; only supply facts and figures (`scripts/04_report_facts.py`,
  `docs/REPORT_OUTLINE.md`). Required content:
  - pipeline design and model choice
  - curves
  - agent search space and policy
  - results table with uncertainty
  - quirk handling
  - confusion-matrix failure analysis
  - what you would do with 10x compute

## Rubric

| Weight | Dimension |
|---|---|
| 15% | Experimental rigor: patient-level stratified split, single test use, seeds, honest uncertainty, identical budgets across rows |
| 25% | Agent design: real search space, genuine feedback loop, auditable log, defences against validation overfitting |
| 20% | FM handling: per-model preprocessing, sound tile→slide aggregation, sensible 20x tiling, efficient caching |
| 20% | Results and analysis: fused beats every single model and the claim is supported; failure analysis matters more than the delta |
| 20% | Engineering: reproducibility, config-driven, typed/documented, modular, no secrets |

## Repository map and commands

- Config: `conf/default.yaml` (headline) and `conf/pilot.yaml` (CPU pilot, layered via `PATHFUSION_CONFIG`).
  Override any key on the CLI: `key.sub=value`.
- Stages:
  - `scripts/00_prepare_data.py`
  - `01_extract_embeddings.py` (GPU)
  - `02_run_agent.py` (resumable via `--run-id`)
  - `03_final_evaluation.py` (once)
  - `04_report_facts.py`
  - `05` metadata/uncertainty, `06` tiling QC, `07` batch probe
- Code: `src/pathfusion/{data,embeddings,models,fusion,training,agent,eval,utils}`.
- Notebooks: `notebooks/kaggle_*` and `colab_*`, generated by `scripts/make_notebooks.py`.
  Edit the generator, not the `.ipynb` files.
- Docs: `docs/WORKLOG.md` (per checkpoint), `RUNBOOK.md`, `NEEDS_YOUR_INPUT.md`, `PILOT.md`,
  `DATA_DECISIONS.md`, `REPORT_OUTLINE.md`.

## Conventions for working in this repo

- **Every commit must show as Verified on GitHub.** Commit with the session's default git
  identity; never override `user.name` / `user.email`, because commits signed with this
  environment's key verify only under that identity. After each push, check
  `commit.verification.verified` on `repos/<owner>/<repo>/commits/<sha>`.
- **Changes go through a pull request into `main`.** Use the existing feature branch; don't
  create extra branches and don't push to `main` directly.
- **Verify before every push:** `ruff check src scripts tests`, `mypy`, and the relevant
  `pytest` tests; for pipeline changes, `python scripts/smoke_e2e.py`. Then confirm CI is
  green on the PR.
- Data-only snapshots (e.g. live agent logs) may use `[skip ci]`.
- Never commit secrets, features (`*.h5`), JPEGs, thumbnails or agent caches (see `.gitignore`).
- Never run stage 3 for exploration. The pilot never runs stage 3.
- Synthetic features (`data/synthetic.py`) are for tests only and never count as results.
- Append a section to `docs/WORKLOG.md` at each checkpoint.
