# Label-Conflicting Near-Duplicate Groups

Derived from `datasets/metadata/duplicate_groups.json` (dHash Hamming threshold ≤ 6).

## Scale

| Measure | Value |
|---|---:|
| Conflicting groups | 42 |
| Images involved | 866 |
| Share of corpus | 14.08% |
| Groups where all members share one vegetable | 6 |
| Groups spanning different vegetables | 36 |
| Median group size | 3 |
| Largest group | 732 |

## Most common conflicting label pairs

| Label A | Label B | Groups | Images |
|---|---|---:|---:|
| Bell Pepper/Dried | Tomato/Old | 6 | 748 |
| Bell Pepper/Dried | Bell Pepper/Ripe | 3 | 740 |
| Bell Pepper/Dried | New Mexico Green Chile/Dried | 3 | 738 |
| Bell Pepper/Old | Tomato/Old | 3 | 742 |
| Bell Pepper/Ripe | New Mexico Green Chile/Dried | 3 | 738 |
| Bell Pepper/Ripe | New Mexico Green Chile/Unripe | 3 | 739 |
| Bell Pepper/Ripe | Tomato/Old | 3 | 739 |
| Chile Pepper/Damaged | New Mexico Green Chile/Dried | 3 | 740 |
| Chile Pepper/Damaged | Tomato/Ripe | 3 | 742 |
| Chile Pepper/Dried | New Mexico Green Chile/Dried | 3 | 737 |
| Chile Pepper/Dried | Tomato/Old | 3 | 738 |
| Chile Pepper/Old | New Mexico Green Chile/Old | 3 | 737 |

## Quality pairs regardless of vegetable

| Quality pair | Groups |
|---|---:|
| Dried ↔ Ripe | 22 |
| Dried ↔ Old | 21 |
| Old ↔ Ripe | 21 |
| Damaged ↔ Ripe | 19 |
| Damaged ↔ Old | 19 |
| Damaged ↔ Dried | 17 |
| Ripe ↔ Unripe | 14 |
| Old ↔ Unripe | 14 |
| Damaged ↔ Unripe | 12 |
| Dried ↔ Unripe | 11 |
| Old ↔ Old | 10 |
| Dried ↔ Dried | 7 |
| Damaged ↔ Damaged | 6 |
| Ripe ↔ Ripe | 6 |
| Unripe ↔ Unripe | 3 |

## Interpretation

Two distinct phenomena are present and must not be conflated:

1. **Genuine visual ambiguity.** `Old` and `Dried` describe overlapping
   states — a shrivelled specimen can reasonably be called either. Groups
   pairing these classes are boundary cases, not annotation errors.
2. **Frames spanning different specimens.** Some 'clusters' are actually
   several similar-looking vegetables photographed in one session. A
   difference in label is then correct: the images look alike because of
   consistent lighting and identical backgrounds, not because they are the
   same object.

The dHash similarity that defines a cluster cannot distinguish these cases.
That is precisely why the splitter keeps clusters whole: it is better to
over-group and lose a handful of training examples than to leak a specimen
across the train/test boundary.

## Consequences for Phase 2

* **A ceiling exists.** 866 images (14.1% of the corpus) are in groups where no model can satisfy every label simultaneously. Perfect accuracy is
  unattainable by construction, independent of model quality.
* **Per-class metrics inherit the noise.** `Dried` and `Old` per-class recall
  will be depressed by boundary cases; this is reported alongside support
  counts and not presented as pure model failure.
* **Error analysis must stratify by this flag.** Phase 2's confusion analysis
  will separate errors on conflict-flagged images from errors on clean images,
  so the reported confusion matrix is not blamed entirely on the model.
* **Relabelling was rejected.** Deciding which label is 'correct' in an
  ambiguous pair requires a domain expert and would silently reshape the
  dataset. Retaining and documenting is the defensible choice; the flag is
  available in `datasets/metadata/duplicate_groups.json` for anyone who
  wants to re-run the analysis with these images excluded.
* **The test split is affected too.** Conflicts were distributed by the
  grouped split, so a conflicting cluster sits entirely in one split; the
  test set therefore contains some of these groups and the reported test
  score carries the same inherent noise. This is stated, not corrected.

## Example groups

See `evaluation/graphs/label_conflict_examples.png` for the six largest
clusters, and `evaluation/metrics/label_conflicts.csv` for every group.
