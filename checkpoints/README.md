# Checkpoints

The trained checkpoint files are not committed to this GitHub repository because several files are larger than 1 GB. They should be distributed through Zenodo, Figshare, institutional storage, or Git LFS.

Download the main trained best-model checkpoint from:

```text
TODO_ZENODO_OR_FIGSHARE_CHECKPOINT_URL
```

Place it here:

```text
checkpoints/flow_noprior_c128_rawamp_best.pt
```

The full checkpoint package prepared for archival upload is located next to this repository:

```text
../boiling-acoustic-video-generation-large-assets/model_checkpoints/
```

It contains:

```text
flow_noprior_c128_rawamp_best.pt
ldm_customvae_c128_rawamp_best.pt
patchgan_edge_ms_best.pt
mmdiffusion_rawcsv_ema060000.pt
mmdiffusion_stft2d_ema060000.pt
dit_flow_c128_rawamp_best.pt
```

See `MODEL_ZOO.md` for the mapping between checkpoint files, model families, and rollout media.
