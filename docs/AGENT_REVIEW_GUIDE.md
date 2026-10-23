# Agent review guide (for the repository owner)

A reading order for checking the fusion-search agent yourself, so you can explain and defend
it, and decide what rules to add to `CLAUDE.md` → "Owner's rules". Each step gives the files to
open, what to look for, and a question to answer in your own words. This is a study aid; don't
copy it into the report.

Run under review: `artifacts/agent/20261023-184822_local_llm/` (headline, Kaggle 2x T4).

## 1. What the agent is allowed to try (≈15 min)

- **Read:** `src/pathfusion/agent/space.py`, then `src/pathfusion/fusion/spec.py` and `pipelines.py`.
- **Look for:**
  - The three families (`early_concat`, `late`, `intermediate`) and their knobs (norm, pca_dim, late_rule, stream_fusion, modality_dropout).
  - The `coverage_specs()` list, which holds the 8 fixed trials.
  - `neighbors()`, which defines the deterministic proposer's local moves.
  - `validate()`, which rejects any configuration outside the space.
- **Answer:** Which brief requirement does each family satisfy? Is anything in the space you
  would remove, or anything missing (e.g. bilinear pooling)?

## 2. How one trial is scored (≈15 min)

- **Read:** `src/pathfusion/agent/evaluator.py` (`CVEvaluator._evaluate`, `paired`).
- **Look for:**
  - Every trial uses the same 5 folds over the 162 development patients (`artifacts/splits/cv_folds_v1.csv`).
  - The test patients are never loaded.
  - The corrected SE of a paired difference uses the Nadeau–Bengio factor.
- **Answer:** Why compare two trials fold by fold instead of comparing their means? Why
  is the plain SE too optimistic with 5 overlapping folds?

## 3. The loop: propose → evaluate → accept or reject → stop (≈20 min)

- **Read:** `src/pathfusion/agent/loop.py`, `AgentLoop.run` (lines 94–145).
- **Look for:**
  - The 3 single-model references come first, and the best one becomes the incumbent.
  - A trial is accepted only if its gain > `acceptance_z` × SE.
  - Patience counts only adaptive trials.
  - The stop rules: budget 24, patience 8, minimum 8 trials.
  - All of these settings live in `conf/default.yaml` → `agent:`.
- **Answer:** In this run 0 of 16 trials were accepted. Is that a failure of the agent, or the
  guardrail working? What would have happened with `acceptance_z: 0`?

## 4. The final pick (≈10 min)

- **Read:** `loop.py`, `_one_se_selection` (lines 150–162), and `spec.complexity()`.
- **Look for:**
  - The pick is the least complex fusion trial within 1 paired SE of the best fusion trial.
  - That is why late fusion with a probability mean (#6, CV 0.882) was picked over early concat with z-score and PCA 128 (#5, CV 0.892).
- **Answer:** Do you agree with "simplest within noise" as the rule? The fused row has to be a
  fusion even when none beats UNI2-h. Is that how you want to present it?

## 5. The LLM and its fallback (≈20 min)

- **Read:**
  - `src/pathfusion/agent/proposers.py`: `SYSTEM_PROMPT`, `LLMProposer._user_prompt`, `propose`.
  - `factory.py`: `resolve_backend`, which picks LLM vs deterministic.
  - Then open `llm_transcript.jsonl` in the run folder.
- **Look for:**
  - The LLM sees the full trial table and must answer in JSON.
  - An invalid or duplicate answer is retried once, then replaced by the deterministic proposer (`source = fallback` in `trials.csv`).
- **Things in this run worth checking yourself:**
  - 4 of the 8 adaptive proposals were duplicates and fell back (#11, #14, #17, #18).
  - Trial #12's reasoning says "the intermediate fusion strategies have not yet been explored",
    but #9 and #10 had already explored them. The reasoning is logged as given, not checked.
- **Answer:** Did the 7B LLM add anything over the deterministic search? Would you require a
  stronger model, or the deterministic proposer only, for the report run?

## 6. The decision log (≈10 min)

- **Read:** `decisions.md` (human-readable), `decisions.jsonl` (machine-readable), `trajectory.png`.
- **Answer:** Could a reviewer reconstruct every move from this log alone? What is missing?

## 7. The fairness control (≈10 min)

- **Read:** `single_search/decisions.md`, and the `equal_budget_single_search` flag.
- **Look for:** the single-model row gets the same agent, budget and selection rule (14 trials). It kept UNI2-h + meta.
- **Answer:** Why does this matter for the "fused beats every single model" claim?

## 8. Did the choice survive to test? (≈15 min)

- **Read:**
  - `artifacts/results/results.md`
  - the `forest_*.png` and `confusion_*.png` figures
  - `error_analysis.csv`
  - `artifacts/within_batch/within_batch.md`
- **Answer:**
  - Are the CV gap (−0.009) and the test gap (+0.003, CI −0.020 to +0.027) consistent?
  - Which classes fail, and do the wrong predictions overlap the pathologist's `Multi_Subtypes`?

## Levers you can restrict (all in `conf/default.yaml`)

| Key | Current | Effect of changing it |
|---|---|---|
| `agent.budget` | 24 | more trials raise the overfitting risk |
| `agent.patience` / `min_trials` | 8 / 8 | when the search may stop |
| `agent.acceptance_z` | 1.0 | how much evidence a new incumbent needs |
| `agent.proposer` | auto | `deterministic` removes the LLM entirely |
| `agent.llm.local_model` | Qwen2.5-7B-Instruct | proposal quality vs GPU memory |
| `agent.equal_budget_single_search` | true | keep this on for a fair comparison |
| `eval.seeds` | 0–4 | more seeds narrow the training-noise spread |

Any change to these is a new agent run. Under the test-once rule, a new run can't be scored
on the test set again; state this if you rerun anything.
