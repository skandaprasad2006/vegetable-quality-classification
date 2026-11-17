# Dataset source record

No archive file is stored here. The images were supplied already extracted in
`Dataset/`, which is preserved untouched as the pristine download. This file
records provenance so the corpus can be re-obtained and so the version actually
used is unambiguous.

## Identity

| Field | Value |
|---|---|
| Name | VegNet: Vegetable Dataset with quality (Unripe, Ripe, Old, Dried and Damaged) |
| Version used | Hugging Face redistribution `Project-AgML/VegNet_quality_classification` |
| Images | 6,150 |
| Format | JPEG, RGB, **all 256×256** |
| Bytes | 131,638,293 (125.5 MiB) |
| Licence | **CC BY 4.0** |
| Vegetables | Bell Pepper, Chile Pepper, New Mexico Green Chile, Tomato |
| Quality classes | Damaged, Dried, Old, Ripe, Unripe |
| Populated cells | 19 of 20 — `Tomato/Dried` is empty |

## Upstream sources

* Mendeley Data, DOI **10.17632/6nxnjbn9w6.1** (v1, published 2022-09-21)
  https://data.mendeley.com/datasets/6nxnjbn9w6/1
* Data in Brief article, DOI **10.1016/j.dib.2022.108657**
  https://doi.org/10.1016/j.dib.2022.108657
* Hugging Face: https://huggingface.co/datasets/Project-AgML/VegNet_quality_classification
  (licence field: `cc-by-4.0`; dataset viewer reports `6.15k rows`, image width 256)

## Known discrepancy

Mendeley and the published paper both state **6,850 images**. The copy used here
is the **6,150-image, 256 px** redistribution: 700 images are absent relative to
the published description, and the originals were downsampled to 256×256.

This is recorded as a limitation, not worked around. Downloading the Mendeley
original to "top up" the corpus was rejected because mixing two differently
preprocessed copies of the same dataset creates near-duplicate leakage across
splits and makes the corpus impossible to describe honestly (see
`Admin/decisions.md` D-004).

## Re-obtaining the data

The Hugging Face redistribution can be re-downloaded with:

```bash
pip install datasets
python -c "from datasets import load_dataset; \
ds = load_dataset('Project-AgML/VegNet_quality_classification'); \
print(ds)"
```

The original full-resolution release is a browser download from the Mendeley
record above (manual acceptance of the CC BY 4.0 terms).

## Attribution

Dataset created by Yogesh Suryawanshi, Kailas Patil and Prawit Chumchu at
Vishwakarma University, Pune, Maharashtra, India. Licensed CC BY 4.0; reuse
requires attribution. See `reports/references.md`.
