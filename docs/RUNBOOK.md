# Runbook: the steps that need you (GPU + accounts)

Everything below runs on Kaggle (free 2x T4). Estimated cost: about 6-10 GPU-hours of the
weekly 30.

## 0. Once: make the code reachable

The notebooks `git clone` branch `claude/keen-noether-926o85` from
`github.com/jkaral/pathology-fusion-AI`, so that branch must be pushed first.

## 1. Stage 1: embeddings (`notebooks/kaggle_01_extract_embeddings.ipynb`)

1. In Kaggle: **Create → New Notebook → File → Import Notebook**, and upload the file.
2. Right panel: Accelerator **GPU T4 x2**, Internet **On**, Secrets `HF_TOKEN` (+ `WANDB_API_KEY`) attached.
3. Run the first cells interactively up to the **2-slide smoke test**. Look at the two QC
   images: green should outline the tissue and blue boxes should sit on tissue only.
   Then **stop the session**.
4. **Save Version → Save & Run All (Commit)**. This runs in the background for up to 12 h.
   - If it times out, rerun it with `SHARD = "0/2"`, then again with `"1/2"`. Finished slides are skipped.
5. When it finishes: **Output** tab → **New Dataset**, named `clwd-embeddings`.

## 2. Stages 2+3: agent + one-time test evaluation (`notebooks/kaggle_02_agent_and_evaluation.ipynb`)

1. Import it the same way. Attach the input dataset `clwd-embeddings`. Use GPU T4 x2,
   Internet On, and the `WANDB_API_KEY` secret.
2. **Save & Run All (Commit)**. The agent uses the local Qwen2.5-7B (free) to propose fusion
   configs, then stage 3 trains every table row with 5 seeds and evaluates on test **once**.
3. Download `results_bundle.tgz` from the Output tab. Unpack it into the repo
   (`artifacts/agent/…`, `artifacts/results/…`) and commit. Then run
   `python scripts/04_report_facts.py` locally (CPU) for the fact sheet.

## 2b. Optional post-hoc study: every slide (`notebooks/kaggle_03_extra_slides_study.ipynb`)

1. Import the notebook. Settings: **GPU T4 x2**, Internet **On**, Secret `HF_TOKEN`, input
   `clwd-embeddings`.
2. **Save & Run All (Commit)**, about 4-5 h. It embeds the 144 further slides of development
   patients (never test patients) and runs `scripts/09_extra_slides_study.py`.
3. Download `extra_slides_bundle.tgz` from the Output tab. It is development CV only and
   doesn't change the headline table.

## 3. The report (yours to write)

Material to draw on, all generated:
- `artifacts/results/report_facts.md`
- `artifacts/results/results.md`
- `artifacts/results/figures/`
- `artifacts/agent/<run>/decisions.md`
- `artifacts/data/data_audit.md`
- `docs/DATA_DECISIONS.md`
- your wandb project `pathology-fusion`

Do not re-run stage 3 to "improve" numbers. The test lock will refuse, and the brief
assesses exactly this.
