# Needs your input

Only items where you are required, or where your call adds real value. Everything else is
done or in progress autonomously (see `docs/WORKLOG.md`).

## Required (only you can do these)

1. **Run the headline GPU pipeline on Kaggle.** UNI2-h and Virchow2 need your HF token, and
   the free GPU is in your account. Steps: `docs/RUNBOOK.md`. The code is pushed now, so the
   notebooks can clone it.
   - `notebooks/kaggle_01_extract_embeddings.ipynb` → save output as dataset `clwd-embeddings`
   - `notebooks/kaggle_02_agent_and_evaluation.ipynb` → download `results_bundle.tgz`
   - Or upload the bundle here and I'll commit it and generate the fact sheet.
2. **Write the technical report** (the brief forbids AI-written prose). The headings mapped
   to the evidence are in `docs/REPORT_OUTLINE.md`.

## Decisions worth your sign-off (defaults are already in place)

| Decision | Default | Why it matters | Where |
|---|---|---|---|
| ID 8377886 (ages 68/69) | one patient | leakage-safe | `data.duplicate_id_policy` |
| 5 patients with conflicting slide labels | label of the imaged slide | image and label always agree; 3 of 5 become Lepidic | `data.conflict_policy` |
| Stain normalisation for the headline run | off | the probe found no batch signal in colour or features so far; final probe pending | `tiling.stain_normalization` |
| LLM search policy on Kaggle | Qwen2.5-7B (free, local) | quality vs. GPU memory | `agent.llm.local_model` |

## Nice to have

- Make the wandb project public, or export its curves, so reviewers can see them.
- The last part of your previous message ("a website videos of ladies with beautiful family
  members who can actually make a dollar … COVID test in the fund") came through garbled,
  probably voice-to-text. I didn't act on it. Please restate what you want.
