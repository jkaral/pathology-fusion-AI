"""Generate the Kaggle/Colab notebooks in notebooks/ (kept in code so they stay in sync).

Usage: python scripts/make_notebooks.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/jkaral/pathology-fusion-AI.git"
BRANCH = "claude/keen-noether-926o85"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def notebook(cells: list[dict]) -> dict:
    return {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3",
                                                        "language": "python"},
                                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


SETUP = f'''
# Clone the pinned branch of the repository and install only what Kaggle lacks.
# NOTE: no "pip -U": upgrading huggingface_hub to 2.x breaks Kaggle's transformers.
import os, subprocess, sys
REPO_DIR = "/kaggle/working/pathology-fusion-AI"
if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone", "--depth", "1", "-b", "{BRANCH}", "{REPO}", REPO_DIR], check=True)
else:
    subprocess.run(["git", "-C", REPO_DIR, "pull", "--ff-only"], check=True)
os.chdir(REPO_DIR)
!pip install -q -r requirements-kaggle.txt
!git log -1 --format="commit %h %s"
'''

SECRETS = '''
# Secrets live in Kaggle "Add-ons -> Secrets" (never in the repo or notebook).
import os
try:
    from kaggle_secrets import UserSecretsClient
    s = UserSecretsClient()
    for name in ("HF_TOKEN", "WANDB_API_KEY"):
        try:
            os.environ[name] = s.get_secret(name)
            print(f"{name}: loaded")
        except Exception:
            print(f"{name}: not set")
except ImportError:
    print("Not on Kaggle: set HF_TOKEN / WANDB_API_KEY in the environment yourself.")
import torch
print("GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
'''


def extraction_notebook() -> dict:
    return notebook([
        md("""
# Stage 1 - tile + embed (UNI2-h, Virchow2, Phikon-v2)

**Settings (right panel):** Accelerator **GPU T4 x2**, Internet **On**, Secrets `HF_TOKEN`
(and optionally `WANDB_API_KEY`) attached.

**How to run the full extraction:** use **Save Version -> Save & Run All (Commit)**, not the
interactive session: commits run in the background for up to 12 h and keep everything in
`/kaggle/working` as the notebook's output. Expect roughly 4-8 h for 203 slides (~218 GB of
JPEG streamed one slide at a time: download -> decode at 20x -> tile -> 3 encoders -> delete).

If one run is not enough, set `SHARD = "0/2"` for the first commit and `"1/2"` for a
second one, then attach both outputs to the stage-2 notebook. Completed slides are
skipped automatically on re-runs.
"""),
        code(SETUP),
        code(SECRETS),
        md("## 1. Disk check and a 2-slide smoke test (look at the QC overlays!)"),
        code('''
import shutil
print("free GB in /kaggle/working:", shutil.disk_usage("/kaggle/working").free / 1e9)
print("free GB in /tmp:", shutil.disk_usage("/tmp").free / 1e9)
os.environ["PATHFUSION_JPG_CACHE"] = "/tmp"           # JPEGs are deleted after each slide
OUT = "/kaggle/working/embeddings"
!python scripts/01_extract_embeddings.py --limit 2 --out-dir {OUT} paths.qc_dir=/kaggle/working/qc
'''),
        code('''
# Green = tissue outline, blue boxes = kept 224px tiles at 20x. Check pen marks / background are excluded.
from IPython.display import Image, display
import glob
for p in sorted(glob.glob("/kaggle/working/qc/*.jpg"))[:2]:
    print(p); display(Image(filename=p, width=700))
'''),
        md("## 2. Full extraction (resumable)"),
        code('''
SHARD = None          # e.g. "0/2" and "1/2" across two commits if 12 h is not enough
shard_arg = f"--shard {SHARD}" if SHARD else ""
!python scripts/01_extract_embeddings.py {shard_arg} --out-dir {OUT} paths.qc_dir=/kaggle/working/qc
'''),
        code('''
!python scripts/01_extract_embeddings.py --summarize --out-dir {OUT}
'''),
        md("""
## 3. Next
After the commit finishes: open the notebook's **Output** tab -> **New Dataset**, name it
`clwd-embeddings`. Attach that dataset to the stage-2 notebook. Also download
`embeddings/extraction_summary.json` and `embeddings/encoders.json` (small) for the report.
"""),
    ])


def agent_eval_notebook() -> dict:
    return notebook([
        md("""
# Stage 2 + 3 - agent fusion search, then the one-time test evaluation

**Settings:** Accelerator **GPU T4 x2** (the local Qwen2.5-7B search policy needs both GPUs),
Internet **On**, input dataset `clwd-embeddings` attached, Secret `WANDB_API_KEY` attached
for the training/validation curves. Use **Save & Run All (Commit)** so nothing is lost.

* Stage 2 sees development patients only (train + val); the test split is never loaded.
* Stage 3 trains every results-table row with 5 seeds and touches test **once**
  (`artifacts/results/TEST_LOCK.json` prevents silent re-runs).
"""),
        code(SETUP),
        code(SECRETS),
        code('''
import glob
cands = sorted({os.path.dirname(p) for p in glob.glob("/kaggle/input/**/WSI-*.h5", recursive=True)})
print(cands)
EMB = "/kaggle/working/all_embeddings"
os.makedirs(EMB, exist_ok=True)
for d in cands:                    # merge shards into one directory of symlinks
    for p in glob.glob(f"{d}/WSI-*.h5"):
        dst = os.path.join(EMB, os.path.basename(p))
        if not os.path.exists(dst):
            os.symlink(p, dst)
print(len(glob.glob(f"{EMB}/*.h5")), "feature files")
'''),
        md("## Stage 2 - agent search (proposer `auto` = local Qwen on GPU, deterministic fallback)"),
        code('''
!python scripts/02_run_agent.py --embeddings-dir {EMB}
!cat artifacts/agent/$(cat artifacts/agent/LATEST)/decisions.md
'''),
        md("## Stage 3 - final evaluation (runs once)"),
        code('''
!python scripts/03_final_evaluation.py --embeddings-dir {EMB}
'''),
        code('''
# Bundle the small, committable artifacts (no features) for download -> add to the repo.
!tar czf /kaggle/working/results_bundle.tgz artifacts/agent artifacts/results
print("Download /kaggle/working/results_bundle.tgz from the Output tab")
'''),
    ])


COLAB_SETUP = f'''
# Colab: features and results live on Google Drive so they survive the runtime.
from google.colab import drive, userdata
drive.mount("/content/drive")
import os, subprocess
WORK = "/content/drive/MyDrive/pathfusion"
os.makedirs(WORK, exist_ok=True)
REPO_DIR = "/content/pathology-fusion-AI"
if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone", "--depth", "1", "-b", "{BRANCH}", "{REPO}", REPO_DIR], check=True)
os.chdir(REPO_DIR)
!pip install -q -r requirements-kaggle.txt
for name in ("HF_TOKEN", "WANDB_API_KEY"):   # Colab: key icon in the left sidebar -> add secret
    try:
        os.environ[name] = userdata.get(name); print(name, "loaded")
    except Exception:
        print(name, "not set")
import torch; print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
'''


def colab_extraction_notebook() -> dict:
    return notebook([
        md("""
# Stage 1 on Colab - tile + embed (UNI2-h, Virchow2, Phikon-v2)

Runtime -> Change runtime type -> **T4 GPU**. Add secrets `HF_TOKEN` (and `WANDB_API_KEY`)
in the key panel. Free Colab sessions can disconnect: just re-run all cells, finished
slides are skipped (features are written to Drive as they complete).
"""),
        code(COLAB_SETUP),
        code('''
os.environ["PATHFUSION_JPG_CACHE"] = "/content/jpg"      # local disk; JPEGs deleted after decoding
OUT = f"{WORK}/embeddings"
!python scripts/01_extract_embeddings.py --limit 2 --out-dir {OUT} paths.qc_dir={WORK}/qc
'''),
        code('''
!python scripts/01_extract_embeddings.py --out-dir {OUT} paths.qc_dir={WORK}/qc
!python scripts/01_extract_embeddings.py --summarize --out-dir {OUT}
'''),
    ])


def colab_agent_eval_notebook() -> dict:
    return notebook([
        md("""
# Stages 2 + 3 on Colab - agent search, then the one-time test evaluation

Single T4 (15 GB): the search policy uses **Qwen2.5-3B-Instruct** in fp16 (the 7B model
needs the two T4s Kaggle provides). Everything else is identical to the Kaggle notebook.
"""),
        code(COLAB_SETUP),
        code('''
EMB = f"{WORK}/embeddings"
OVR = f"paths.agent_dir={WORK}/agent paths.results_dir={WORK}/results agent.llm.local_model=Qwen/Qwen2.5-3B-Instruct"
!python scripts/02_run_agent.py --embeddings-dir {EMB} {OVR}
'''),
        code('''
!python scripts/03_final_evaluation.py --embeddings-dir {EMB} {OVR}
!python scripts/04_report_facts.py --embeddings-dir {EMB} {OVR}
'''),
    ])


def main() -> None:
    out = ROOT / "notebooks"
    out.mkdir(exist_ok=True)
    (out / "kaggle_01_extract_embeddings.ipynb").write_text(json.dumps(extraction_notebook(), indent=1) + "\n")
    (out / "kaggle_02_agent_and_evaluation.ipynb").write_text(json.dumps(agent_eval_notebook(), indent=1) + "\n")
    (out / "colab_01_extract_embeddings.ipynb").write_text(json.dumps(colab_extraction_notebook(), indent=1) + "\n")
    (out / "colab_02_agent_and_evaluation.ipynb").write_text(
        json.dumps(colab_agent_eval_notebook(), indent=1) + "\n")
    print("wrote", sorted(p.name for p in out.glob("*.ipynb")))


if __name__ == "__main__":
    main()
