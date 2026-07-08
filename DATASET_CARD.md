# Dataset Card

## Dataset

Representative public release of the corrected CSV acoustic-video pool-boiling dataset.

## Source

The full dataset was acquired on a transparent pool-boiling facility with synchronized video and acoustic measurements.

Per case:

- 100-fps RGB video
- 1 MHz CSV acoustic waveform
- static background image
- heating ROI mask

## Released Subset

This release includes representative training and validation subsets and the full held-out test split.

```text
Train: 54 / 107 cases = 50.47%
Validation: 8 / 13 cases = 61.54%
Test: 14 / 14 cases = 100%
Selection policy: heat-flux-stratified train/validation subsets; full test release
```

The released training subset covers:

```text
0, 5, 20, 30, 40, 100, 200, 300, 400, 500, 600, 700, 800, 895, 900 kW/m^2
```

## File Structure

```text
examples/train_50pct_representative/
examples/val_50pct_representative/
examples/test_full/
examples/manifests/train_50pct_representative.jsonl
examples/manifests/val_50pct_representative.jsonl
examples/manifests/test_full.jsonl
```

## Preprocessing

- CSV waveforms are read directly without WAV conversion.
- No per-window peak normalization is applied.
- Acoustic amplitudes are clipped to the acquisition range of +/-10 V in the final model.
- 8-frame video chunks correspond to 80 ms at 100 fps.
- Each 8-frame chunk is paired with the co-temporal 80 ms acoustic segment.

## Intended Use

This release is intended for reviewer testing, code verification, demonstration, and direct test-set inference/evaluation. The full manuscript training benchmark still requires the complete corrected training dataset and released checkpoints.

## Limitations

- The train and validation releases are representative subsets, not the complete train/validation dataset.
- The full held-out test split is released.
- The full reported training benchmark requires the complete corrected training dataset and trained checkpoints.
