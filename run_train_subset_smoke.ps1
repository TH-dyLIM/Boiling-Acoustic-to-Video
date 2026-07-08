param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path -LiteralPath "cache/release_subset/train")) {
    Write-Host "Cache not found. Preparing cache first..." -ForegroundColor Yellow
    & $Python scripts/precompute_residual_tensor_cache.py `
        --config configs/release_train_subset_smoke.json `
        --cache_root cache/release_subset `
        --manifest train=examples/manifests/train_50pct_representative.jsonl `
        --manifest val=examples/manifests/val_50pct_representative.jsonl `
        --manifest test=examples/manifests/test_full.jsonl `
        --overwrite
    if ($LASTEXITCODE -ne 0) {
        throw "precompute_residual_tensor_cache failed"
    }
}

& $Python scripts/train_flow_residual.py --config configs/release_train_subset_smoke.json
if ($LASTEXITCODE -ne 0) {
    throw "train_flow_residual smoke test failed"
}

Write-Host "Done. Smoke-test training outputs written to outputs/train_subset_smoke" -ForegroundColor Green
