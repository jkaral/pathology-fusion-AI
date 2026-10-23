"""Additional slides of development patients, for the post-hoc training-data study.

The headline protocol uses one slide per patient (lowest available ``WSI_ID``). Of the
386 slides in the HF release, 183 are further slides of the same patients; 144 belong to
development (train + val) patients. All come from the same CLWD release (same hospital,
scanner and pathologist labels), so no external data enters the study.

Rules, so the study cannot touch the test set or change what is being scored:

* only slides of **development** patients are listed; test patients' slides never are;
* the headline slide of each patient is excluded (it is already in the headline cohort);
* each extra slide carries **its own** label from ``slide_labels`` (consistent with the
  ``selected_slide`` policy: the label always belongs to the imaged slide);
* extra slides become extra *training* bags of their patient's fold. Validation folds are
  still scored on the headline slide, so CV numbers stay comparable with the agent run.

An extra slide is represented as a pseudo-patient row ``"<patient_id>/<WSI_ID>"`` with the
parent's age and sex, so ``FeatureStore`` and ``DataContext`` work unchanged.
"""

from __future__ import annotations

import pandas as pd

SEP = "/"


def extra_slide_table(cohort: pd.DataFrame, split: pd.DataFrame, manifest: pd.DataFrame,
                      classes: list[str]) -> pd.DataFrame:
    """One row per additional available slide of a development patient.

    Columns: ``parent_id, WSI_ID, label, label_idx, size_bytes, split``.
    """
    sp = split.assign(patient_id=split["patient_id"].astype(str)).set_index("patient_id")["split"]
    available = dict(zip(manifest["WSI_ID"], manifest["size_bytes"]))
    rows = []
    for _, r in cohort.iterrows():
        pid = str(r["patient_id"])
        part = sp[pid]
        if part == "test":
            continue
        for item in str(r["slide_labels"]).split("|"):
            wsi, label = item.split(":", 1)
            if wsi == r["WSI_ID"] or wsi not in available:
                continue
            rows.append({"parent_id": pid, "WSI_ID": wsi, "label": label,
                         "label_idx": classes.index(label), "size_bytes": int(available[wsi]),
                         "split": part})
    out = pd.DataFrame(rows, columns=["parent_id", "WSI_ID", "label", "label_idx", "size_bytes", "split"])
    return out.sort_values("WSI_ID", key=lambda s: s.str.split("-").str[1].astype(int)).reset_index(drop=True)


def extra_cohort_rows(cohort: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    """Pseudo-patient cohort rows (``<patient_id>/<WSI_ID>``) for the extra slides."""
    parents = cohort.assign(patient_id=cohort["patient_id"].astype(str)).set_index("patient_id")
    rows = parents.loc[extra["parent_id"]].reset_index()
    rows["patient_id"] = [f"{p}{SEP}{w}" for p, w in zip(extra["parent_id"], extra["WSI_ID"])]
    rows["WSI_ID"] = extra["WSI_ID"].to_numpy()
    rows["label"] = extra["label"].to_numpy()
    rows["label_idx"] = extra["label_idx"].to_numpy()
    rows["size_bytes"] = extra["size_bytes"].to_numpy()
    return rows[cohort.columns]


def extras_by_parent(extra: pd.DataFrame) -> dict[str, list[str]]:
    """Parent patient id -> pseudo-patient ids of its extra slides."""
    out: dict[str, list[str]] = {}
    for p, w in zip(extra["parent_id"], extra["WSI_ID"]):
        out.setdefault(str(p), []).append(f"{p}{SEP}{w}")
    return out
