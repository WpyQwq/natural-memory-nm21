# Train both v6 routers, then score every router and print the dominance verdict.
# This is chain_v6 without the encoder wait: it is used once the feature bank has
# been verified complete by verify_feature_bank.py.
param(
    [int]$Steps = 100000,
    [string[]]$Configs = @('v2_512_v6', 'xl512_v6')
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
if ($LASTEXITCODE -ne 0) { throw "v6 training failed with exit code $LASTEXITCODE" }

$old = 'H:\Memory\dynamic_memory_lab\checkpoints'
& $Python -m V2_dpskw.eval_router_v5 `
    --train-file data/router_training_v6/train.jsonl `
    --eval-file data/router_training_v6/eval.jsonl `
    --feature-cache $CacheDir `
    --run "V2-128 deployed(v3)=$old\natural_memory_v2_qwen_router_entities\memory_router_v2.pt" `
    --run "V2-512 v3 best=$old\natural_memory_v2_router_512\router_best.pt" `
    --run "V2-512 v3 final=$old\natural_memory_v2_router_512\memory_router_v2.pt" `
    --run "V2-512 v6 best=$ForkRoot\checkpoints\router_v6_v2_512\router_best.pt" `
    --run "V2-512 v6 final=$ForkRoot\checkpoints\router_v6_v2_512\memory_router_v2.pt" `
    --run "XL-512 v6 best=$ForkRoot\checkpoints\router_v6_xl512\router_best.pt" `
    --run "XL-512 v6 final=$ForkRoot\checkpoints\router_v6_xl512\memory_router_xl.pt" `
    --output router_scorecard_v6.json --markdown router_scorecard_v6.md
if ($LASTEXITCODE -ne 0) { throw "scorecard failed with exit code $LASTEXITCODE" }

& $Python -m V2_dpskw.verdict_router_v6 `
    --scorecard router_scorecard_v6.json `
    --candidate "XL-512 v6 best" --candidate "XL-512 v6 final" `
    --candidate "V2-512 v6 best" --candidate "V2-512 v6 final" `
    --baseline-prefix "V2-128 deployed" --baseline-prefix "V2-512 v3" `
    --output router_verdict_v6.json --markdown router_verdict_v6.md
Write-Output ("verdict exit code: {0} (0 = complete dominance, 2 = some axis failed)" -f $LASTEXITCODE)
Write-Output ("[{0}] train+score complete" -f (Get-Date -Format 'HH:mm:ss'))
