# Nature Reviewer Quickstart

This quickstart reproduces a minimal inference run using the provided representative data subset.

## 1. Install

```bash
conda create -n boiling-a2v python=3.10 -y
conda activate boiling-a2v
pip install -r requirements.txt
```

## 2. Download Checkpoint

Download the trained checkpoint from:

```text
TODO_ZENODO_OR_FIGSHARE_CHECKPOINT_URL
```

Save it as:

```text
checkpoints/flow_noprior_c128_rawamp_best.pt
```

## 3. Prepare Cache

PowerShell:

```powershell
.\run_prepare_example_cache.ps1 -Python python -Overwrite
```

or Python:

```bash
python scripts/precompute_residual_tensor_cache.py \
  --config configs/release_train_subset_smoke.json \
  --cache_root cache/release_subset \
  --manifest train=examples/manifests/train_50pct_representative.jsonl \
  --manifest val=examples/manifests/val_50pct_representative.jsonl \
  --manifest test=examples/manifests/test_full.jsonl \
  --overwrite
```

Expected result:

```text
saved: 76
cache/release_subset/train/*.pt
cache/release_subset/val/*.pt
cache/release_subset/test/*.pt
```

## 4. Run Inference

PowerShell:

```powershell
.\run_infer_quick_demo.ps1 -Python python
```

or Python:

```bash
python scripts/sample_flow_residual.py \
  --checkpoint checkpoints/flow_noprior_c128_rawamp_best.pt \
  --manifest examples/manifests/quick_demo_3cases.jsonl \
  --cache_dir cache/release_subset/test \
  --prior_mode none \
  --output_dir outputs/quick_demo_predictions \
  --full_video_rollout \
  --num_inference_steps 30 \
  --cfg_scale 1.0 \
  --save_fps 10 \
  --mixed_precision bf16
```

Expected output folders:

```text
outputs/quick_demo_predictions/gifs/
outputs/quick_demo_predictions/pred_mp4/
outputs/quick_demo_predictions/samples_summary.json
```

## 5. Optional Smoke-test Training

This does not reproduce the manuscript model; it only verifies that the training code executes on the public subset.

```powershell
.\run_train_subset_smoke.ps1 -Python python
```

## Notes

- The manuscript model was trained on the full corrected dataset.
- The public representative subset is provided for reviewer testing and code verification.
- The final model uses no video-derived nucleation prior at inference.
- The current implementation is not real time at 100 fps because 30 Euler steps are used for sampling.
