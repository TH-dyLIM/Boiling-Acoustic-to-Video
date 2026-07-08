# Model Zoo and Reviewer Assets

This file maps the manuscript model families to their trained checkpoints and generated rollout media.

The repository includes MP4 rollouts directly under:

```text
assets/paper_selected_videos/
assets/model_rollouts/
```

Large checkpoint files are packaged separately for archival upload under:

```text
../boiling-acoustic-video-generation-large-assets/model_checkpoints/
```

Before public release, upload those checkpoint files to Zenodo, Figshare, institutional storage, or Git LFS, and replace the placeholder links in `README.md` and `checkpoints/README.md`.

## Main Models

| Model ID | Family | Checkpoint | Rollout location | Notes |
|---|---|---|---|---|
| `flow_noprior_c128_rawamp` | Conditional flow matching | `flow_noprior_c128_rawamp_best.pt` | `assets/model_rollouts/flow_noprior_c128_rawamp/` | Main no-prior model; best by FVD in the final comparison |
| `ldm_customvae_c128_rawamp` | Latent diffusion | `ldm_customvae_c128_rawamp_best.pt` | `assets/model_rollouts/ldm_customvae_c128_rawamp/` | Best LDM baseline using a custom VAE |
| `patchgan_edge_ms` | PatchGAN residual generator | `patchgan_edge_ms_best.pt` | `assets/model_rollouts/patchgan_edge_ms/` | Best adversarial sharpening baseline |
| `mmdiffusion_rawcsv_ema060k` | DDPM-style MM-Diffusion | `mmdiffusion_rawcsv_ema060000.pt` | `assets/model_rollouts/mmdiffusion_rawcsv_ema060k/` | Raw-CSV MM-Diffusion baseline |
| `mmdiffusion_stft2d_ema060k` | DDPM-style MM-Diffusion | `mmdiffusion_stft2d_ema060000.pt` | `assets/paper_selected_videos/MM-Diffusion_customized_260515_stft2d/` | STFT2D MM-Diffusion media used in manuscript figure preparation |
| `dit_flow_c128_rawamp` | DiT-style flow model | `dit_flow_c128_rawamp_best.pt` | `assets/model_rollouts/dit_flow_c128_rawamp/` | Transformer-backbone flow-matching ablation |

## Reported Metrics

The compact model comparison table used in the manuscript is stored at:

```text
results/benchmark_matrix.xlsx
```

The media/checkpoint manifest is stored at:

```text
assets/model_zoo_manifest.csv
assets/paper_selected_videos/paper_selected_video_manifest.csv
```

The final selected model in the manuscript is:

```text
flow_noprior_c128_rawamp
```

It is a no-video-prior residual conditional flow-matching model with a pixel-space 3D U-Net backbone. Inference uses only the raw CSV acoustic waveform, static background image, and heating ROI mask.

