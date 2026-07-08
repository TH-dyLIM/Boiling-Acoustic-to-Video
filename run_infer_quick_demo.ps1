param(
    [string]$Python = "python",
    [string]$Checkpoint = "checkpoints/flow_noprior_c128_rawamp_best.pt",
    [int]$SampleSteps = 30
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path -LiteralPath $Checkpoint)) {
    throw "Checkpoint not found: $Checkpoint. Download it from the archive URL in checkpoints/README.md."
}
if (-not (Test-Path -LiteralPath "cache/release_subset/test")) {
    throw "Cache not found. Run .\\run_prepare_example_cache.ps1 first."
}

& $Python scripts/sample_flow_residual.py `
    --checkpoint $Checkpoint `
    --manifest examples/manifests/quick_demo_3cases.jsonl `
    --cache_dir cache/release_subset/test `
    --prior_mode none `
    --output_dir outputs/quick_demo_predictions `
    --full_video_rollout `
    --num_inference_steps $SampleSteps `
    --cfg_scale 1.0 `
    --save_fps 10 `
    --mixed_precision bf16
if ($LASTEXITCODE -ne 0) {
    throw "sample_flow_residual failed"
}

Write-Host "Done. Outputs written to outputs/quick_demo_predictions" -ForegroundColor Green
