# NM2 整体记忆测试电池（可用于任何包：NM2.1 合并包 / 原版 NM2）
#
# 跑完整记忆能力测试：
#   A 写入→读取→作答（多类别，跨类别正确率）
#   B 零字面重叠改写泛化（16 条同形候选）
#   C 同形候选大样本 + 未知问题泄漏
#   D 重启持久化（关闭进程后记忆是否还在）
#
# 输出文件名由 -Tag 派生，避免同一目录下跑多个包时互相覆盖。
# 百分比一律为百分比。
param(
    [string]$Package = 'qwen3_5_4b_natural_memory_v2_1',
    [string]$Label = 'NM2.1',
    [string]$Tag = 'nm2_1',
    [int]$PerCategory = 10,
    [int]$OverlapCases = 48,
    [string]$Blends = '0.5',
    [switch]$SkipA,
    [switch]$SkipB,
    [switch]$SkipC,
    [switch]$SkipD
)

$ErrorActionPreference = 'Stop'
$Python = 'C:\Users\Administrator\miniconda3\envs\LLM\python.exe'
$env:PYTHONPATH = 'H:\Memory'
$env:PYTHONIOENCODING = 'utf-8'
Set-Location 'H:\Memory\V2_dpskw'

$fA_json = "${Tag}_e2e.json";          $fA_md = "${Tag}_e2e.md"
$fB_json = "${Tag}_critical_e2e.json"; $fB_md = "${Tag}_critical_e2e.md"
$fC_json = "${Tag}_runtime_e2e.json";  $fC_md = "${Tag}_runtime_e2e.md"
$fD_json = "${Tag}_restart.json"
$fS_json = "${Tag}_battery_summary.json"

$results = [ordered]@{}
function Step($name, $block) {
    Write-Output ("[{0}] === {1} ===" -f (Get-Date -Format HH:mm:ss), $name)
    try { $script:results[$name] = & $block }
    catch { Write-Output ("  FAILED: {0}" -f $_.Exception.Message); $script:results[$name] = @{ error = $_.Exception.Message } }
}

if (-not $SkipA) {
    Step 'A_end_to_end' {
        & $Python -m V2_dpskw.eval_end_to_end_memory --package $Package `
            --eval-file data/router_training_v6/eval.jsonl `
            --per-category $PerCategory --facts 6 --max-new-tokens 48 `
            --router "$Label=" `
            --output $fA_json --markdown $fA_md 2>&1 | Select-Object -Last 3
        (Get-Content $fA_json -Raw | ConvertFrom-Json)
    }
}

if (-not $SkipB) {
    Step 'B_zero_overlap_paraphrase' {
        & $Python -m V2_dpskw.eval_router_critical_e2e --package $Package --cases 16 `
            --router "$Label=" `
            --output $fB_json --markdown $fB_md 2>&1 | Select-Object -Last 3
        (Get-Content $fB_json -Raw | ConvertFrom-Json)
    }
}

if (-not $SkipC) {
    Step 'C_same_shape_and_unknown' {
        & $Python -m V2_dpskw.eval_runtime_zero_overlap_e2e --package $Package `
            --cases $OverlapCases --blends $Blends --prior-scales 1.0 `
            --output $fC_json --markdown $fC_md 2>&1 | Select-Object -Last 12
        (Get-Content $fC_json -Raw | ConvertFrom-Json)
    }
}

if (-not $SkipD) {
    Step 'D_restart_persistence' {
        & $Python -m V2_dpskw.test_natural_memory_v2_restart --model-path $Package `
            --report $fD_json --max-new-tokens 16 2>&1 | Select-Object -Last 4
        if (Test-Path $fD_json) { (Get-Content $fD_json -Raw | ConvertFrom-Json) } else { @{ error = 'no report' } }
    }
}

$summary = [ordered]@{
    package       = $Package
    label         = $Label
    tag           = $Tag
    built_at      = (Get-Date).ToString('s')
    per_category  = $PerCategory
    overlap_cases = $OverlapCases
    results       = $results
}
$summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $fS_json -Encoding UTF8
Write-Output ("wrote {0}" -f $fS_json)
