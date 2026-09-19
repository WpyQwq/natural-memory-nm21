# Train 512-dim routers on the corrected v6 dataset (87,155 train / 21,920 eval).
#
# Identical data, features, sampling and optimisation for both runs, so the
# comparison isolates the router architecture:
#   * v2_512_v6  —— the V2 architecture (4.74M params), the control;
#   * xl512_v6   —— the new MemoryRouterXL (7.90M params), the candidate.
#   * xl512_top1_v6 —— optional Top-1-weighted XL recipe (available on demand).
param(
    [int]$Steps = 100000,
    [int]$EvalInterval = 500,
    [int]$BatchSize = 64,
    [double]$GpuMemoryGb = 9,
    [string]$SamplingMode = 'family_sqrt',
    [string[]]$Configs = @('v2_512_v6', 'xl512_v6'),
    [switch]$Resume
)

$ErrorActionPreference = 'Stop'

$Python = 'C:\Users\Administrator\miniconda3\envs\LLM\python.exe'
$ForkRoot = 'H:\Memory\V2_dpskw'
$Parent = 'H:\Memory'
$TrainFile = 'data/router_training_v6/train.jsonl'
$EvalFile = 'data/router_training_v6/eval.jsonl'
$FeatureCache = 'H:\Memory\nm_cache\nm_router_v6\feature_cache'
$ModelPath = 'qwen3_5_4b_natural_memory_v2'

$Presets = @{
    'v2_512_v6' = [pscustomobject]@{
        Label = 'v2_512_v6'
        OutputDir = 'checkpoints/router_v6_v2_512'
        Extra = @()
        Args = @('--arch v2', '--router-dim 512', '--num-heads 8')
    }
    'xl512_v6' = [pscustomobject]@{
        Label = 'xl512_v6'
        OutputDir = 'checkpoints/router_v6_xl512'
        Extra = @()
        Args = @(
            '--arch xl', '--router-dim 512', '--num-heads 8',
            '--encoder-layers 2', '--encoder-hidden 512',
            '--pair-blocks 1', '--pair-hidden 512', '--pair-expansion 2', '--pair-dropout 0.05',
            '--policy-layers 2', '--policy-hidden 512'
        )
    }
    # Same storage budget as the deployed production router (128 dims = 512 bytes
    # per record, ~2.0M params).  Comparing a 512-dim router against a 128-dim one
    # on address bytes is a category error, so the like-for-like dominance claim has
    # to be made at the deployed geometry.
    'v2_128_v6' = [pscustomobject]@{
        Label = 'v2_128_v6'
        OutputDir = 'checkpoints/router_v6_v2_128'
        Extra = @()
        Args = @('--arch v2', '--router-dim 128', '--num-heads 8')
    }
    'xl128_v6' = [pscustomobject]@{
        Label = 'xl128_v6'
        OutputDir = 'checkpoints/router_v6_xl128'
        Extra = @()
        Args = @(
            '--arch xl', '--router-dim 128', '--num-heads 8',
            '--encoder-layers 2', '--encoder-hidden 512',
            '--pair-blocks 1', '--pair-hidden 512', '--pair-expansion 2', '--pair-dropout 0.05',
            '--policy-layers 2', '--policy-hidden 512'
        )
    }
    'xl512_top1_v6' = [pscustomobject]@{
        Label = 'xl512_top1_v6'
        OutputDir = 'checkpoints/router_v6_xl512_top1'
        Extra = @('--margin-loss-weight 0.6', '--margin 0.2', '--learning-rate 1.5e-4')
        Args = @(
            '--arch xl', '--router-dim 512', '--num-heads 8',
            '--encoder-layers 2', '--encoder-hidden 512',
            '--pair-blocks 1', '--pair-hidden 512', '--pair-expansion 2', '--pair-dropout 0.05',
            '--policy-layers 2', '--policy-hidden 512'
        )
    }
    # Latency-parity variant: keeps the parts that drive quality (MLP encoder,
    # interaction features, multi-layer policy heads) and drops the residual pair
    # trunk, which is the single most expensive block per candidate.
    'xl512_lean_v6' = [pscustomobject]@{
        Label = 'xl512_lean_v6'
        OutputDir = 'checkpoints/router_v6_xl512_lean'
        Extra = @()
        Args = @(
            '--arch xl', '--router-dim 512', '--num-heads 8',
            '--encoder-layers 2', '--encoder-hidden 512',
            '--pair-blocks 0', '--pair-hidden 512',
            '--policy-layers 2', '--policy-hidden 512'
        )
    }
}

if (-not (Test-Path -LiteralPath $Python)) { throw "Python not found: $Python" }
foreach ($required in @($TrainFile, $EvalFile)) {
    if (-not (Test-Path -LiteralPath (Join-Path $ForkRoot $required))) { throw "Missing dataset: $required" }
}
if (-not (Test-Path -LiteralPath (Join-Path $FeatureCache 'manifest.json'))) { throw "Missing feature bank: $FeatureCache" }

$env:PYTHONPATH = $Parent
Set-Location $ForkRoot

$common = @(
    '-m V2_dpskw.train_router_v5',
    "--train-file $TrainFile",
    "--eval-file $EvalFile",
    "--feature-cache $FeatureCache",
    "--model-path $ModelPath",
    "--steps $Steps",
    "--batch-size $BatchSize",
    "--eval-interval $EvalInterval",
    "--sampling-mode $SamplingMode",
    "--gpu-memory-gb $GpuMemoryGb",
    '--log-every 500'
)

$started = @()
foreach ($name in $Configs) {
    if (-not $Presets.ContainsKey($name)) { throw "unknown config: $name" }
    $run = $Presets[$name]
    $outputDir = Join-Path $ForkRoot $run.OutputDir
    New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

    $resumePath = ''
    if ($Resume) {
        $latest = Get-ChildItem -Path $outputDir -Filter 'router_step_*.pt' -ErrorAction SilentlyContinue |
            Sort-Object Name -Descending | Select-Object -First 1
        if ($latest) { $resumePath = $latest.FullName }
    }

    $argumentLine = ($common + @("--label $($run.Label)", "--output-dir $($run.OutputDir)"))
    if ($resumePath) {
        $argumentLine += @("--resume `"$resumePath`"")
        Write-Output ("resuming {0} from {1}" -f $run.Label, (Split-Path $resumePath -Leaf))
    } else {
        $argumentLine += @('--overwrite-metrics')
    }
    $argumentLine = ($argumentLine + $run.Extra + $run.Args) -join ' '

    $stdout = Join-Path $outputDir 'training_stdout.log'
    $stderr = Join-Path $outputDir 'training_stderr.log'
    Write-Output ("starting {0}: {1}" -f $run.Label, $argumentLine)
    $process = Start-Process -FilePath $Python -WorkingDirectory $ForkRoot `
        -ArgumentList $argumentLine -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr -WindowStyle Hidden -PassThru -Wait
    Write-Output ("{0} exited with code {1}" -f $run.Label, $process.ExitCode)
    $started += [pscustomobject]@{ label = $run.Label; output_dir = $run.OutputDir; exit_code = $process.ExitCode }
}

$started | ConvertTo-Json -Compress
