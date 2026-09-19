param(
    [double]$RefreshSeconds = 0.5
)

$TrainingRoot = 'H:\Memory\V2_dpskw\checkpoints\natural_memory_v2_router_v3_100k'
$MetricsPath = Join-Path $TrainingRoot 'metrics.jsonl'
$SummaryPath = Join-Path $TrainingRoot 'memory_router_large_training.json'
$TargetSteps = 100000

$Host.UI.RawUI.WindowTitle = 'Natural Memory v2 Router V3 - Live Monitor'

function Get-TrainingProcess {
    Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object {
            $_.CommandLine -and
            $_.CommandLine -like '*train_memory_router_large*' -and
            $_.CommandLine -like '*router_v3*'
        } |
        Select-Object -First 1
}

function Read-Events {
    if (-not (Test-Path -LiteralPath $MetricsPath)) {
        return @()
    }
    $events = @()
    foreach ($line in (Get-Content -LiteralPath $MetricsPath -Tail 300 -ErrorAction SilentlyContinue)) {
        try {
            $events += ($line | ConvertFrom-Json -ErrorAction Stop)
        } catch {
            # Ignore the one line that may be being appended right now.
        }
    }
    return $events
}

function Get-GpuStatus {
    $raw = & nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>$null
    if (-not $raw) {
        return $null
    }
    $parts = $raw -split ',' | ForEach-Object { $_.Trim() }
    if ($parts.Count -lt 4) {
        return $null
    }
    return [pscustomobject]@{
        Name = $parts[0]
        UsedMiB = $parts[1]
        TotalMiB = $parts[2]
        Utilization = $parts[3]
    }
}

while ($true) {
    $procInfo = Get-TrainingProcess
    $events = Read-Events
    $lastTrain = @($events | Where-Object { $_.event -eq 'train' } | Select-Object -Last 1)[0]
    $lastEval = @($events | Where-Object { $_.event -in @('eval', 'eval_resume') } | Select-Object -Last 1)[0]
    $gpu = Get-GpuStatus

    Clear-Host
    Write-Host 'Natural Memory v2 - Router V3 Live Monitor' -ForegroundColor Cyan
    Write-Host ('Updated: {0}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))
    Write-Host ('Output: {0}' -f $TrainingRoot)
    Write-Host ''

    if ($procInfo) {
        $proc = Get-Process -Id $procInfo.ProcessId -ErrorAction SilentlyContinue
        $ram = if ($proc) { '{0:N2} GB' -f ($proc.WorkingSet64 / 1GB) } else { 'N/A' }
        $cpu = if ($proc) { '{0:N1}s' -f $proc.CPU } else { 'N/A' }
        Write-Host ('Status: RUNNING   PID={0}   RAM={1}   CPU={2}' -f $procInfo.ProcessId, $ram, $cpu) -ForegroundColor Green
    } else {
        Write-Host 'Status: training process not found' -ForegroundColor Yellow
    }

    if ($lastTrain) {
        $percent = [math]::Min(100, 100 * [double]$lastTrain.step / $TargetSteps)
        Write-Host ('Progress: {0}/{1} ({2:N1}%)' -f $lastTrain.step, $TargetSteps, $percent)
        Write-Host ('Loss={0:N5}  Route={1:N5}  Need={2:N6}  Hop={3:N6}  LR={4:E3}' -f [double]$lastTrain.loss, [double]$lastTrain.route_loss, [double]$lastTrain.need_loss, [double]$lastTrain.hop_loss, [double]$lastTrain.lr)
    } else {
        Write-Host 'Progress: waiting for training log...' -ForegroundColor DarkYellow
    }

    if ($lastEval) {
        Write-Host ''
        Write-Host ('Last evaluation: step {0}' -f $lastEval.step) -ForegroundColor Magenta
        Write-Host ('Top1={0:P2}  RecallAt3={1:P2}  MRR={2:P2}' -f [double]$lastEval.route_top1, [double]$lastEval.route_recall_at3, [double]$lastEval.route_mrr)
        Write-Host ('NeedF1={0:P2}  Abstention={1:P2}  Hop={2:P2}' -f [double]$lastEval.need_f1, [double]$lastEval.abstention_accuracy, [double]$lastEval.hop_accuracy)
    }

    if ($gpu) {
        Write-Host ''
        Write-Host ('GPU: {0}   VRAM: {1}/{2} MiB   Utilization: {3}%' -f $gpu.Name, $gpu.UsedMiB, $gpu.TotalMiB, $gpu.Utilization) -ForegroundColor DarkCyan
    }

    Write-Host ''
    Write-Host 'Press Ctrl+C to close the monitor. Training will not be stopped.' -ForegroundColor DarkGray

    if (-not $procInfo -and (Test-Path -LiteralPath $SummaryPath)) {
        Write-Host ''
        Write-Host 'Training complete. Final report is available.' -ForegroundColor Green
        Write-Host 'Monitor is staying on the final result. Press Ctrl+C to close.' -ForegroundColor DarkGray
        Start-Sleep -Milliseconds ([int]([math]::Max(100, $RefreshSeconds * 1000)))
        continue
    }
    Start-Sleep -Milliseconds ([int]([math]::Max(100, $RefreshSeconds * 1000)))
}
