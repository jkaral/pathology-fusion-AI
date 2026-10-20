"""Metadata loading, data-quirk handling and one-slide-per-patient cohort construction.

The cohort is the single table every later stage reads: one row per patient with the
selected slide, the patient-level label and the covariates (age, sex).

Quirks handled here (see ``audit_quirks`` and ``docs/DATA_DECISIONS.md``):

1. ``SampleNumber`` 8377886 appears with Age 68 and 69 (two specimens a year apart).
2. Five patients carry different growth-pattern labels on different slides.
3. 22 of the 408 listed slides are absent from the Hugging Face release; 6 patients
   have no available slide at all and cannot be used.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pathfusion.config import Config

SEX_CODES = {"Female": 0, "Male": 1}


def load_slides(cfg: Config) -> pd.DataFrame:
    """Slide-level table: brief metadata + availability in the HF release + extra HF columns."""
    slides = pd.read_csv(cfg.paths.resolve("metadata_csv"))
    slides.columns = [c.strip() for c in slides.columns]
    slides["SampleNumber"] = slides["SampleNumber"].astype(str)
    slides["wsi_num"] = slides["WSI_ID"].str.extract(r"WSI-(\d+)", expand=False).astype(int)
    slides = slides.rename(columns={cfg.data.label_column: "label"})

    manifest_path = cfg.paths.resolve("hf_manifest_csv")
    manifest = pd.read_csv(manifest_path)
    slides = slides.merge(manifest[["WSI_ID", "size_bytes"]], on="WSI_ID", how="left")
    slides["available"] = slides["size_bytes"].notna()

    hf_path = cfg.paths.resolve("metadata_hf_csv")
    if Path(hf_path).exists():
        hf = pd.read_csv(hf_path)[["WSI_ID", "Multi_Subtypes", "Benchmark_Label_7class"]]
        slides = slides.merge(hf, on="WSI_ID", how="left")
        mismatch = (slides["label"] != slides["Benchmark_Label_7class"]).sum()
        if mismatch:
            raise ValueError(f"{mismatch} labels differ between brief CSV and HF CSV")
        slides = slides.drop(columns="Benchmark_Label_7class")

    unknown = set(slides["label"]) - set(cfg.data.classes)
    if unknown:
        raise ValueError(f"Labels not in configured class list: {unknown}")
    return slides.sort_values("wsi_num").reset_index(drop=True)


def assign_patient_ids(slides: pd.DataFrame, policy: str) -> pd.DataFrame:
    """Add ``patient_id``. ``merge`` keys on SampleNumber; ``split`` separates ages."""
    slides = slides.copy()
    if policy == "merge":
        slides["patient_id"] = slides["SampleNumber"]
    elif policy == "split":
        n_ages = slides.groupby("SampleNumber")["Age"].transform("nunique")
        slides["patient_id"] = np.where(
            n_ages > 1,
            slides["SampleNumber"] + "_age" + slides["Age"].astype(str),
            slides["SampleNumber"],
        )
    else:
        raise ValueError(f"Unknown duplicate_id_policy {policy!r}")
    return slides


@dataclass
class QuirkReport:
    duplicate_ids: pd.DataFrame
    conflicting_patients: pd.DataFrame
    missing_slides: pd.DataFrame
    patients_without_slides: pd.DataFrame
    sex_inconsistent: list[str]


def audit_quirks(slides: pd.DataFrame) -> QuirkReport:
    """Detect every metadata irregularity programmatically (nothing is hard-coded)."""
    by_id = slides.groupby("SampleNumber")
    dup = slides[by_id["Age"].transform("nunique") > 1]
    conflict = slides[by_id["label"].transform("nunique") > 1]
    missing = slides[~slides["available"]]
    no_avail = slides[~by_id["available"].transform("any")]
    sex_bad = [k for k, g in by_id if g["Sex"].nunique() > 1]
    cols = ["SampleNumber", "WSI_ID", "Age", "Sex", "label", "available"]
    return QuirkReport(
        duplicate_ids=dup[cols + ["Multi_Subtypes"] if "Multi_Subtypes" in dup else cols],
        conflicting_patients=conflict[cols + ["Pathological_Diagnosis"]],
        missing_slides=missing[cols],
        patients_without_slides=no_avail.groupby("SampleNumber")
        .agg(label=("label", "first"), n_slides=("WSI_ID", "size"))
        .reset_index(),
        sex_inconsistent=sex_bad,
    )


def build_cohort(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, QuirkReport]:
    """Return (cohort, excluded, quirk report). Cohort has exactly one row per patient."""
    slides = assign_patient_ids(load_slides(cfg), cfg.data.duplicate_id_policy)
    report = audit_quirks(slides)
    if report.sex_inconsistent:
        raise ValueError(f"Sex differs within patient(s): {report.sex_inconsistent}")
    if cfg.data.slide_selection != "lowest_available_wsi_id":
        raise ValueError(f"Unknown slide_selection {cfg.data.slide_selection!r}")

    rows, excluded = [], []
    for pid, grp in slides.groupby("patient_id", sort=False):
        grp = grp.sort_values("wsi_num")
        avail = grp[grp["available"]]
        all_labels = grp["label"].tolist()
        conflict = len(set(all_labels)) > 1
        if avail.empty:
            excluded.append({"patient_id": pid, "reason": "no slide in HF release",
                             "label": all_labels[0]})
            continue
        sel = avail.iloc[0]
        label = _resolve_label(cfg.data.conflict_policy, sel["label"], all_labels)
        if label is None:
            excluded.append({"patient_id": pid, "reason": "conflicting slide labels",
                             "label": "|".join(sorted(set(all_labels)))})
            continue
        rows.append({
            "patient_id": pid,
            "SampleNumber": sel["SampleNumber"],
            "WSI_ID": sel["WSI_ID"],
            "Age": int(sel["Age"]),       # age at the selected specimen
            "Sex": sel["Sex"],
            "sex_code": SEX_CODES[sel["Sex"]],
            "label": label,
            "label_idx": cfg.data.classes.index(label),
            "diagnosis": sel["Pathological_Diagnosis"],
            "multi_subtypes": sel.get("Multi_Subtypes", ""),
            "n_slides_listed": len(grp),
            "n_slides_available": len(avail),
            "slide_labels": "|".join(f"{w}:{lab}" for w, lab in zip(grp["WSI_ID"], all_labels)),
            "label_conflict": conflict,
            "size_bytes": int(sel["size_bytes"]),
        })
    cohort = pd.DataFrame(rows).sort_values("patient_id").reset_index(drop=True)
    return cohort, pd.DataFrame(excluded), report


def _resolve_label(policy: str, selected: str, all_labels: list[str]) -> str | None:
    if len(set(all_labels)) == 1 or policy == "selected_slide":
        return selected
    if policy == "exclude":
        return None
    if policy == "majority":
        counts = pd.Series(all_labels).value_counts()
        top = counts[counts == counts.max()].index.tolist()
        return selected if selected in top else top[0]
    raise ValueError(f"Unknown conflict_policy {policy!r}")


def covariates(cohort: pd.DataFrame, age_mean: float, age_std: float) -> np.ndarray:
    """Model covariates [age_z, sex] with age standardised by *training* statistics."""
    age_z = (cohort["Age"].to_numpy(float) - age_mean) / age_std
    return np.stack([age_z, cohort["sex_code"].to_numpy(float)], axis=1).astype(np.float32)
