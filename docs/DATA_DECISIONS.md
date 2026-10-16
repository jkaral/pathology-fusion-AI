# Data decisions

All choices are implemented in `src/pathfusion/data/metadata.py`, switchable in
`conf/default.yaml`, and their effect is regenerated in `artifacts/data/data_audit.md`
by `scripts/00_prepare_data.py`.

## 1. `SampleNumber` 8377886 appears twice (Age 68 and 69)

Slides WSI-35/36 are listed at age 68 and WSI-103/104 at 69. All are female and Papillary;
the `Multi_Subtypes` field differs (Papillary&Lepidic vs. Papillary&Lepidic&Acinar), which
fits two specimens from one patient a year apart, though a recycled ID cannot be ruled out.

**Default `duplicate_id_policy: merge`** (one patient). It is the leakage-safe choice:
if the two are the same person, splitting them could put the same patient in train and test,
which the brief calls disqualifying. If they are different people, merging only costs one
training example. The selected slide is WSI-35, and age is taken from that specimen
(68). `split` is available as an ablation.

## 2. Five patients have different labels on different slides

| patient | slides |
|---|---|
| 8225322 | WSI-347 Lepidic, WSI-348 Papillary |
| 8240634 | WSI-372 Acinar (IA), WSI-373 In situ (AIS) |
| 8245975 | WSI-380 Acinar, WSI-381 Solid |
| 8248415 | WSI-398 Lepidic, WSI-399 Acinar |
| 8248805 | WSI-401 Lepidic, WSI-402 Acinar |

The labels are per-slide annotations, and adenocarcinoma is heterogeneous, so both labels
can be correct for their own slide. The reduced protocol images one slide per patient.
**Default `conflict_policy: selected_slide`**: the patient label is the label of the slide
that is actually imaged, so the image and its target always agree. A "dominant pattern"
label would sometimes pair an image with a pattern that is absent from it.

Consequence to note: with lowest-`WSI_ID` selection, three of the five resolve to Lepidic.
Alternatives: `exclude` drops the five patients (198 remain); `majority` keeps the most
frequent label, with ties going to the selected slide.

## 3. Slides missing from the Hugging Face release (found during the audit)

`CLWD.csv` lists 408 slides, but the HF dataset contains 386 JPEGs (see
`data/hf_file_manifest.csv`, built from the HF file tree). For 16 patients, other slides
remain. **Six patients have no available slide** (8388318, 8219606, 8221926, 8226738,
8238589, 8243660; 3 Acinar, 2 Lepidic, 1 Papillary) and are excluded. The usable cohort is
**203 patients**.

## 4. Slide selection

Lowest numeric `WSI_ID` *among slides present in the release*. The rule is deterministic
and blind to labels and images.

## 5. Split

Allocation is per class: `n_test = round(0.2 n)`, `n_val = max(1, round(0.1 n))`, and the
rest go to train, after a seeded shuffle. Every class appears in every split (Cribriform:
10/2/3). Result: **140 / 22 / 41** patients. `split_v1.csv` and its `.sha256` are committed,
and every stage verifies the checksum and the absence of leakage on load.

## 6. Which metadata file

`data/CLWD.csv` is the brief's file, and its labels are used. `data/CLWD_hf.csv` is the
richer HF release. The labels are identical (checked on load). Only its `Multi_Subtypes`
column is used, for error analysis: whether a "wrong" prediction is a pattern the
pathologist recorded as present.
