param(
    [string]$Python = "python",
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$argsList = @(
    "scripts/precompute_residual_tensor_cache.py",
    "--config", "configs/release_train_subset_smoke.json",
    "--cache_root", "cache/release_subset",
    "--manifest", "train=examples/manifests/train_50pct_representative.jsonl",
    "--manifest", "val=examples/manifests/val_50pct_representative.jsonl",
    "--manifest", "test=examples/manifests/test_full.jsonl"
)
if ($Overwrite) {
    $argsList += "--overwrite"
}

& $Python @argsList
if ($LASTEXITCODE -ne 0) {
    throw "precompute_residual_tensor_cache failed"
}

Write-Host "Done. Cache written to cache/release_subset" -ForegroundColor Green
