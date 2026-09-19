# Chain: wait for the v5 feature encoder, then train and score automatically.
#
#   1. wait for the encoder process to exit;
#   2. verify the encoder wrote its real manifest (not the smoke placeholder);
#   3. train the V2-512 control and the XL-512 candidate on the identical v5 data;
#   4. score every router (old production, old v3-trained, new v5-trained) on the
#      frozen 21,920-episode v5 eval and write the percentage scorecard.
param(
    [Parameter(Mandatory = $true)][int]$EncoderPid,
    [int]$Steps = 100000,
    [int]$PollSeconds = 30
)

$ErrorActionPreference = 'Stop'
$Python = 'C:\Users\Administrator\miniconda3\envs\LLM\python.exe'
$ForkRoot = 'H:\Memory\V2_dpskw'
$CacheDir = 'H:\Memory\nm_cache\nm_router_v5\feature_cache'
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
& pwsh -NoProfile -File (Join-Path $ForkRoot 'run_router_v5.ps1') -Steps $Steps
if ($LASTEXITCODE -ne 0) { throw "v5 training failed with exit code $LASTEXITCODE" }

# --- 4. score ---------------------------------------------------------------
$env:PYTHONPATH = 'H:\Memory'
Set-Location $ForkRoot
$old = 'H:\Memory\dynamic_memory_lab\checkpoints'
& $Python -m V2_dpskw.eval_router_v5 `
    --run "V2-128 deployed(v3)=$old\natural_memory_v2_qwen_router_entities\memory_router_v2.pt" `
    --run "V2-512 v3 best=$old\natural_memory_v2_router_512\router_best.pt" `
    --run "V2-512 v3 final=$old\natural_memory_v2_router_512\memory_router_v2.pt" `
    --run "V2-512 v5 best=$ForkRoot\checkpoints\router_v5_v2_512\router_best.pt" `
    --run "V2-512 v5 final=$ForkRoot\checkpoints\router_v5_v2_512\memory_router_v2.pt" `
    --run "XL-512 v5 best=$ForkRoot\checkpoints\router_v5_xl512\router_best.pt" `
    --run "XL-512 v5 final=$ForkRoot\checkpoints\router_v5_xl512\memory_router_xl.pt" `
    --output router_scorecard_v5.json --markdown router_scorecard_v5.md
Write-Output ("[{0}] chain complete" -f (Get-Date -Format 'HH:mm:ss'))
