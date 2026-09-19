# Chain: wait for the v6 feature encoder, then train and score automatically.
#
#   1. wait for the encoder process to exit;
#   2. verify the encoder wrote its real manifest;
#   3. train the V2-512 control and the XL-512 candidate on the identical v6 data;
#   4. score every router (old production 128d, old v3-trained 512s, new v6 512s)
#      on the frozen 21,920-episode v6 eval and write the percentage scorecard.
param(
    [Parameter(Mandatory = $true)][int]$EncoderPid,
    [int]$Steps = 100000,
    [int]$PollSeconds = 30
)

$ErrorActionPreference = 'Stop'
$Python = 'C:\Users\Administrator\miniconda3\envs\LLM\python.exe'
$ForkRoot = 'H:\Memory\V2_dpskw'
$CacheDir = 'H:\Memory\nm_cache\nm_router_v6\feature_cache'
$Manifest = Join-Path $CacheDir 'manifest.json'

Write-Output ("[{0}] waiting for encoder pid {1}" -f (Get-Date -Format 'HH:mm:ss'), $EncoderPid)
while (Get-Process -Id $EncoderPid -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds $PollSeconds
}
Write-Output ("[{0}] encoder exited" -f (Get-Date -Format 'HH:mm:ss'))

$manifest = Get-Content -LiteralPath $Manifest -Raw | ConvertFrom-Json
if (-not $manifest.PSObject.Properties.Name.Contains('encoding')) {
    throw "manifest has no 'encoding' section: the encoder did not finish cleanly"
}
if ([int]$manifest.text_count -lt 2000000) {
    throw "encoder produced only $($manifest.text_count) texts"
}
Write-Output ("manifest ok: texts={0} dtype={1} grouping={2}" -f $manifest.text_count, $manifest.dtype, $manifest.encoding.grouping)

# --- 3. train ---------------------------------------------------------------
& pwsh -NoProfile -File (Join-Path $ForkRoot 'run_router_v6.ps1') -Steps $Steps
if ($LASTEXITCODE -ne 0) { throw "v6 training failed with exit code $LASTEXITCODE" }

# --- 4. score ---------------------------------------------------------------
$env:PYTHONPATH = 'H:\Memory'
Set-Location $ForkRoot
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

# --- 5. verdict: per-axis dominance against the best baseline ---------------
& $Python -m V2_dpskw.verdict_router_v6 `
    --scorecard router_scorecard_v6.json `
    --candidate "XL-512 v6 best" --candidate "XL-512 v6 final" `
    --candidate "V2-512 v6 best" --candidate "V2-512 v6 final" `
    --baseline-prefix "V2-128 deployed" --baseline-prefix "V2-512 v3" `
    --output router_verdict_v6.json --markdown router_verdict_v6.md
$verdictCode = $LASTEXITCODE
Write-Output ("verdict exit code: {0} (0 = complete dominance, 2 = some axis failed)" -f $verdictCode)
Write-Output ("[{0}] chain complete" -f (Get-Date -Format 'HH:mm:ss'))
