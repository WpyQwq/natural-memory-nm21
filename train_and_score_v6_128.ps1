# Train the 128-dim routers (deployed production geometry) and score everything.
#
# Why: the 512-dim routers lost the cost axes only against the *deployed* router,
# which stores 512 bytes per record against their 2048.  That is a different
# storage budget, not a better router.  Training at 128 dims makes the comparison
# like-for-like: identical address size, identical ~2.0M parameter class, and the
# question becomes whether the new data/labels dominate the shipped router.
param(
    [int]$Steps = 100000,
    [string[]]$Configs = @('v2_128_v6', 'xl128_v6')
)

$ErrorActionPreference = 'Stop'
$Python = 'C:\Users\Administrator\miniconda3\envs\LLM\python.exe'
$ForkRoot = 'H:\Memory\V2_dpskw'
$CacheDir = 'H:\Memory\nm_cache\nm_router_v6\feature_cache'

$env:PYTHONPATH = 'H:\Memory'
Set-Location $ForkRoot

$verification = & $Python -m V2_dpskw.verify_feature_bank --cache-dir $CacheDir --no-update-manifest 2>&1 | Out-String
if ($verification -notmatch '"complete": true') {
    throw "feature bank is not complete; refusing to train:`n$verification"
}
Write-Output 'feature bank: COMPLETE (verified by full scan)'

& (Join-Path $ForkRoot 'run_router_v6.ps1') -Steps $Steps -Configs $Configs
if ($LASTEXITCODE -ne 0) { throw "128-dim training failed with exit code $LASTEXITCODE" }

$old = 'H:\Memory\dynamic_memory_lab\checkpoints'
& $Python -m V2_dpskw.eval_router_v5 `
    --train-file data/router_training_v6/train.jsonl `
    --eval-file data/router_training_v6/eval.jsonl `
    --feature-cache $CacheDir `
    --run "V2-128 deployed(v3)=$old\natural_memory_v2_qwen_router_entities\memory_router_v2.pt" `
    --run "V2-128 v6 best=$ForkRoot\checkpoints\router_v6_v2_128\router_best.pt" `
    --run "V2-128 v6 final=$ForkRoot\checkpoints\router_v6_v2_128\memory_router_v2.pt" `
    --run "XL-128 v6 best=$ForkRoot\checkpoints\router_v6_xl128\router_best.pt" `
    --run "XL-128 v6 final=$ForkRoot\checkpoints\router_v6_xl128\memory_router_xl.pt" `
    --run "XL-512 v6 best=$ForkRoot\checkpoints\router_v6_xl512\router_best.pt" `
    --run "V2-512 v6 best=$ForkRoot\checkpoints\router_v6_v2_512\router_best.pt" `
    --output router_scorecard_v6_128.json --markdown router_scorecard_v6_128.md
if ($LASTEXITCODE -ne 0) { throw "scorecard failed with exit code $LASTEXITCODE" }

& $Python -m V2_dpskw.verdict_router_v6 `
    --scorecard router_scorecard_v6_128.json `
    --candidate "V2-128 v6 best" --candidate "V2-128 v6 final" `
    --candidate "XL-128 v6 best" --candidate "XL-128 v6 final" `
    --baseline-prefix "V2-128 deployed" `
    --output router_verdict_v6_128.json --markdown router_verdict_v6_128.md
Write-Output ("verdict exit code: {0} (0 = complete dominance vs the deployed router, 2 = some axis failed)" -f $LASTEXITCODE)
Write-Output ("[{0}] 128-dim train+score complete" -f (Get-Date -Format 'HH:mm:ss'))
