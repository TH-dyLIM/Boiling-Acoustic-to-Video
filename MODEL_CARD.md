# Model Card

## Model

No-video-prior residual conditional flow-matching U-Net for acoustic-to-video pool-boiling visualization.

## Intended Use

The model generates plausible pool-boiling video rollouts from external acoustic measurements and deployable visual conditions. It is intended for research on non-invasive boiling visualization and model benchmarking.

## Inputs

- Raw-amplitude CSV acoustic waveform
- Static background image
- Heating ROI mask

The final model does not use video-derived nucleation priors, future frames, heat-flux labels, HTC labels, or boiling-regime labels at inference.

## Output

Generated RGB boiling video. The model predicts residual video and composes it with the measured static background.

## Architecture

- Pixel-space 3D U-Net
- Conditional flow-matching objective
- Raw waveform 1D audio encoder
- Dense background and ROI conditioning
- Base channels: 128
- Audio tokens: 24
- Conditioning dimension: 384
- Time embedding dimension: 192

## Training

- Optimizer: AdamW
- Learning rate: 7e-5
- Weight decay: 1e-4
- Batch size: 1
- Max training steps: 55,000
- Best checkpoint step: 40,000
- Precision: bf16
- GPU: NVIDIA GeForce RTX 3090 24 GB
- Validation interval: 1,000 steps
- Best checkpoint criterion: validation distribution score

## Inference

- Euler sampling steps: 30
- CFG scale: 1.0
- Chunk length: 8 frames at 100 fps
- Resolution: 128 x 128

## Limitations

- The model generates plausible stochastic boiling visualizations, not deterministic pixel-exact reconstructions.
- The released model was trained on a single experimental facility.
- External facility validation remains future work.
- Current 30-step Euler sampling is not real time at 100 fps on an RTX 3090.

