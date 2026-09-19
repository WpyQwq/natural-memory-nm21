param(
    [int]$Steps = 100000,
    [int]$EvalInterval = 500,
    [int]$CheckpointInterval = 5000,
    [int]$GpuMemoryGb = 9
)

$ErrorActionPreference = 'Stop'

$Python = 'C:\Users\Administrator\miniconda3\envs\LLM\python.exe'
$WorkDir = 'H:\Memory'
$OutputDir = 'H:\Memory\V2_dpskw\checkpoints\natural_memory_v2_router_512'
$FeatureCacheDir = 'H:\Memory\V2_dpskw\checkpoints\natural_memory_v2_router_v3\feature_cache'
$TrainFile = 'V2_dpskw/data/router_training_v3/train.jsonl'
$EvalFile = 'V2_dpskw/data/router_training_v3/eval.jsonl'

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python executable not found: $Python"
}
if (-not (Test-Path -LiteralPath (Join-Path $WorkDir $TrainFile))) {
    throw "Training file not found: $TrainFile"
}
if (-not (Test-Path -LiteralPath (Join-Path $WorkDir $EvalFile))) {
    throw "Evaluation file not found: $EvalFile"
}
if (-not (Test-Path -LiteralPath $FeatureCacheDir)) {
    throw "Feature cache not found: $FeatureCacheDir"
}
if ($Steps -lt 1) { throw 'Steps must be positive.' }
if ($EvalInterval -ne 500) { throw 'This production protocol requires EvalInterval=500.' }
if ($CheckpointInterval -lt 1) { throw 'CheckpointInterval must be positive.' }

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$existing = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'python.exe' -and
    $_.CommandLine -match 'train_memory_router_large' -and
    $_.CommandLine -match 'natural_memory_v2_router_512'
}
if ($existing) {
    throw "A 512d router training process is already running: $($existing.ProcessId -join ', ')"
}

$stdout = Join-Path $OutputDir 'training_stdout.log'
$stderr = Join-Path $OutputDir 'training_stderr.log'
$argumentLine = @(
    '-m V2_dpskw.train_memory_router_large',
    '--train-file V2_dpskw/data/router_training_v3/train.jsonl',
    '--eval-file V2_dpskw/data/router_training_v3/eval.jsonl',
    '--model-path V2_dpskw/qwen3_5_4b_natural_memory_v2',
    '--output-dir V2_dpskw/checkpoints/natural_memory_v2_router_512',
    '--feature-cache-dir V2_dpskw/checkpoints/natural_memory_v2_router_v3/feature_cache',
    '--router-dim 512',
    '--num-heads 8',
    '--max-hops 3',
    '--gpu-memory-gb ' + $GpuMemoryGb,
    '--encode-batch-size 1',
    '--steps ' + $Steps,
    '--batch-size 64',
    '--eval-batch-size 128',
    '--eval-interval ' + $EvalInterval,
    '--checkpoint-interval ' + $CheckpointInterval
) -join ' '

$process = Start-Process -FilePath $Python -WorkingDirectory $WorkDir `
    -ArgumentList $argumentLine -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr -WindowStyle Hidden -PassThru

[pscustomobject]@{
    pid = $process.Id
    router_dim = 512
    router_parameters = 4741902
    steps = $Steps
    eval_interval = $EvalInterval
    checkpoint_interval = $CheckpointInterval
    output_dir = $OutputDir
    feature_cache = $FeatureCacheDir
} | ConvertTo-Json -Compress
