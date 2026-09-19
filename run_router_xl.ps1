# Launch MemoryRouterXL training runs in the fork.
#
# Order matters: the 512-dim run goes first because router_dim == the stored
# address size per memory record, so 512 matches the existing 512-dim baseline
# exactly (same per-record storage cost) and isolates router-internal capacity.
#
# Every run reuses:
#   * the frozen router_training_v3 dataset (same train/eval as the 512 baseline),
#   * the frozen Qwen feature cache (Qwen is never loaded; no re-encoding),
#   * the identical evaluation protocol of train_memory_router_large.
#
# Usage:
#   pwsh -File H:\Memory\V2_dpskw\run_router_xl.ps1
#   pwsh -File H:\Memory\V2_dpskw\run_router_xl.ps1 -Configs xl512
#   pwsh -File H:\Memory\V2_dpskw\run_router_xl.ps1 -Configs xl512 -Resume
param(
    [int]$Steps = 100000,
    [int]$EvalInterval = 500,
    [int]$BatchSize = 64,
    [double]$GpuMemoryGb = 9,
    [string[]]$Configs = @('xl512'),
    [switch]$Resume
)

$ErrorActionPreference = 'Stop'

$Python = 'C:\Users\Administrator\miniconda3\envs\LLM\python.exe'
$ForkRoot = 'H:\Memory\V2_dpskw'
$Parent = 'H:\Memory'
$TrainFile = 'data/router_training_v3/train.jsonl'
$EvalFile = 'data/router_training_v3/eval.jsonl'
$CacheDir = 'checkpoints/router_shared/feature_cache'
$ModelPath = 'qwen3_5_4b_natural_memory_v2'

# Architecture presets.  ``xl512`` keeps router_dim = 512 and num_heads = 8, i.e.
# the exact address geometry of the V2 512 baseline, and spends the extra
# capacity inside the encoder, the pair trunk and the policy heads.
$Presets = @{
    'xl512' = [pscustomobject]@{
        Label = 'router_xl_512'
        OutputDir = 'checkpoints/router_xl_512'
        Args = @(
            '--router-dim 512', '--num-heads 8',
            '--encoder-layers 2', '--encoder-hidden 512',
            '--pair-blocks 1', '--pair-hidden 512', '--pair-expansion 2', '--pair-dropout 0.05',
            '--policy-layers 2', '--policy-hidden 512'
        )
    }
    'xl1024' = [pscustomobject]@{
        Label = 'router_xl_1024'
        OutputDir = 'checkpoints/router_xl_1024'
        Args = @(
            '--router-dim 1024', '--num-heads 16',
            '--encoder-layers 2', '--encoder-hidden 1024',
            '--pair-blocks 1', '--pair-hidden 1024', '--pair-expansion 2', '--pair-dropout 0.05',
            '--policy-layers 2', '--policy-hidden 512'
        )
    }
    'xl2048' = [pscustomobject]@{
        Label = 'router_xl_2048'
        OutputDir = 'checkpoints/router_xl_2048'
        Args = @(
            '--router-dim 2048', '--num-heads 16',
            '--encoder-layers 2', '--encoder-hidden 2048',
            '--pair-blocks 1', '--pair-hidden 2048', '--pair-expansion 2', '--pair-dropout 0.05',
            '--policy-layers 2', '--policy-hidden 512'
        )
    }
}

if (-not (Test-Path -LiteralPath $Python)) { throw "Python not found: $Python" }
foreach ($required in @($TrainFile, $EvalFile)) {
    if (-not (Test-Path -LiteralPath (Join-Path $ForkRoot $required))) { throw "Missing data file: $required" }
}
if (-not (Test-Path -LiteralPath (Join-Path $ForkRoot $CacheDir))) { throw "Missing feature cache: $CacheDir" }

$env:PYTHONPATH = $Parent
Set-Location $ForkRoot

# Refuse to start if the frozen feature cache does not match the frozen data:
# otherwise the trainer would silently load the 4B model and re-encode.
$cacheCheck = & $Python -m V2_dpskw.check_router_cache 2>&1 | Out-String
if ($cacheCheck -notmatch 'CACHE HIT') {
    throw "feature cache check failed:`n$cacheCheck"
}
Write-Output 'feature cache: HIT'

$common = @(
    '-m V2_dpskw.train_memory_router_xl',
    "--train-file $TrainFile",
    "--eval-file $EvalFile",
    "--feature-cache-dir $CacheDir",
    "--model-path $ModelPath",
    "--steps $Steps",
    "--batch-size $BatchSize",
    "--eval-interval $EvalInterval",
    "--gpu-memory-gb $GpuMemoryGb",
    '--log-every 100'
)

$started = @()
foreach ($name in $Configs) {
    if (-not $Presets.ContainsKey($name)) { throw "unknown config: $name" }
    $run = $Presets[$name]
    $outputDir = Join-Path $ForkRoot $run.OutputDir
    New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
    $runArgs = $run.Args

    $resumePath = ''
    if ($Resume) {
        $latest = Get-ChildItem -Path $outputDir -Filter 'router_step_*.pt' -ErrorAction SilentlyContinue |
            Sort-Object Name -Descending | Select-Object -First 1
        if ($latest) {
            $resumePath = $latest.FullName
            Write-Output ("resuming {0} from {1}" -f $run.Label, $latest.Name)
        }
    }

    $argumentLine = ($common + @("--label $($run.Label)", "--output-dir $($run.OutputDir)"))
    if ($resumePath) {
        $argumentLine += @("--resume `"$resumePath`"")
    } else {
        $argumentLine += @('--overwrite-metrics')
    }
    $argumentLine = ($argumentLine + $runArgs) -join ' '

    $stdout = Join-Path $outputDir 'training_stdout.log'
    $stderr = Join-Path $outputDir 'training_stderr.log'
    Write-Output ("starting {0}: {1}" -f $run.Label, $argumentLine)
    $process = Start-Process -FilePath $Python -WorkingDirectory $ForkRoot `
        -ArgumentList $argumentLine -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr -WindowStyle Hidden -PassThru -Wait
    Write-Output ("{0} exited with code {1}" -f $run.Label, $process.ExitCode)
    $started += [pscustomobject]@{ label = $run.Label; output_dir = $run.OutputDir; exit_code = $process.ExitCode; stdout = $stdout; stderr = $stderr }
}

$started | ConvertTo-Json -Compress
