# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

[CmdletBinding()]
param(
    [ValidateSet("Submit", "Status", "Fetch", "Resume", "Cancel")]
    [string]$Action = "Submit",
    [string]$Server = "192.168.13.190",
    [string]$SshUser = "root",
    [ValidateRange(1, 65535)]
    [int]$SshPort = 22,
    [string]$Container = "qwen3_parallel_nightly",
    [string]$Model = "/models/Qwen3-30B-A3B",
    [ValidateSet("moe", "dense", "auto")]
    [string]$ModelKind = "moe",
    [ValidateSet("quick", "boundary")]
    [string]$Matrix = "quick",
    [ValidateSet("eager", "aclgraph")]
    [string]$ExecutionMode = "aclgraph",
    [string]$ExpectedVllmVersion = "",
    [string]$ExpectedVllmAscendVersion = "",
    [string]$SpecFile = "",
    [int[]]$InputTokens = @(128, 512, 2048),
    [int[]]$OutputTokens = @(64, 256),
    [int[]]$Concurrency = @(1, 4, 16, 32),
    [ValidateRange(1, 100000)]
    [int]$RequestsPerPoint = 64,
    [ValidateRange(1, 20)]
    [int]$Repetitions = 3,
    [ValidateRange(1, 1024)]
    [int]$WarmupRequests = 4,
    [ValidateRange(1, 100000)]
    [int]$ProfileNumRequests = 16,
    [ValidateRange(1, 100000)]
    [int]$ProfileConcurrency = 4,
    [ValidateRange(1, 1048576)]
    [int]$ProfileInputTokens = 512,
    [ValidateRange(1, 1048576)]
    [int]$ProfileOutputTokens = 64,
    [string]$RunId = "",
    [string]$OutputDir = "D:\vllm-parallel-bench",
    [string]$RemoteProject = "/home/vllm/l00977701/pipeline_parallel",
    [string]$RemoteRunRoot = "/home/vllm/l00977701/runtime/parallel_bench_runs",
    [ValidateRange(1, 21600)]
    [int]$MaxWaitSeconds = 21600,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$target = "${SshUser}@${Server}"
$toolDir = $PSScriptRoot
$selectorScript = Join-Path (Split-Path $PSScriptRoot -Parent) "find_idle_npu.sh"
$requiredFiles = @(
    "benchmark_remote.py",
    "experiment_schema.py",
    "resource_scheduler.py",
    "workload_streaming.py",
    "profiler_collect.py",
    "metrics_analyzer.py",
    "report_generator.py"
)

foreach ($program in @("ssh", "scp")) {
    if (-not (Get-Command $program -ErrorAction SilentlyContinue)) {
        throw "$program was not found. Install Windows OpenSSH Client."
    }
}

function Invoke-Ssh {
    param([Parameter(Mandatory = $true)][string]$Command)
    & ssh -p $SshPort $target $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Remote command failed with exit code $LASTEXITCODE."
    }
}

if ($Action -ne "Submit" -and [string]::IsNullOrWhiteSpace($RunId)) {
    throw "-RunId is required for $Action."
}
if ([string]::IsNullOrWhiteSpace($RunId)) {
    $RunId = "parallel-" + (Get-Date -Format "yyyyMMdd-HHmmss")
}
if ($RunId -notmatch "^[A-Za-z0-9][A-Za-z0-9_.-]*$") {
    throw "RunId contains unsafe characters."
}

$remoteTool = "$RemoteProject/profile_tool"
$remoteRun = "$RemoteRunRoot/$RunId"
$remoteSpec = "$remoteRun/spec.json"
$remoteController = "$remoteTool/benchmark_remote.py"

switch ($Action) {
    "Status" {
        $statusCommand = "test -s '$remoteRun/state.json' && cat '$remoteRun/state.json' " +
            "|| { echo 'state not found'; exit 2; }; " +
            "pid=''; test -s '$remoteRun/controller.pid' && pid=`$(cat '$remoteRun/controller.pid'); " +
            "case `"`$pid`" in (*[!0-9]*|`"`") echo 'CONTROLLER_ALIVE=false';; " +
            "(*) if kill -0 `"`$pid`" 2>/dev/null; then echo 'CONTROLLER_ALIVE=true'; " +
            "else echo 'CONTROLLER_ALIVE=false'; fi;; esac"
        Invoke-Ssh $statusCommand
        exit 0
    }
    "Fetch" {
        New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
        & scp -P $SshPort -r "${target}:$remoteRun" $OutputDir
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to fetch $RunId."
        }
        Write-Host "Downloaded: $(Join-Path $OutputDir $RunId)"
        Write-Host "Report: $(Join-Path (Join-Path $OutputDir $RunId) 'report.html')"
        exit 0
    }
    "Cancel" {
        Invoke-Ssh "touch '$remoteRun/CANCEL'"
        Write-Host "Cancellation requested. The controller will stop its own current service at the next checkpoint."
        exit 0
    }
    "Resume" {
        $launch = "cd '$remoteTool' && test -s '$remoteSpec' && " +
            "rm -f '$remoteRun/CANCEL' && " +
            "nohup python3 '$remoteController' --run-dir '$remoteRun' --spec '$remoteSpec' " +
            ">> '$remoteRun/controller.log' 2>&1 < /dev/null & echo `$! > '$remoteRun/controller.pid'"
        Invoke-Ssh $launch
        Write-Host "Resumed detached run: $RunId"
        exit 0
    }
}

foreach ($file in $requiredFiles) {
    $path = Join-Path $toolDir $file
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Missing bundled file: $path"
    }
}
if (-not (Test-Path -LiteralPath $selectorScript -PathType Leaf)) {
    throw "Missing bundled selector: $selectorScript"
}

$spec = [ordered]@{
    schema_version = 1
    server = [ordered]@{
        host = $Server
        ssh_user = $SshUser
        remote_root = $RemoteRunRoot
    }
    container = [ordered]@{
        name = $Container
        expected_vllm_version = $ExpectedVllmVersion
        expected_vllm_ascend_version = $ExpectedVllmAscendVersion
    }
    model = [ordered]@{
        name = [IO.Path]::GetFileName($Model.TrimEnd("/"))
        path = $Model
        kind = $ModelKind
        served_name = "parallel-bench"
        max_model_len = 4096
    }
    matrix = [ordered]@{ preset = $Matrix; cases = @() }
    workload = [ordered]@{
        input_tokens = $InputTokens
        output_tokens = $OutputTokens
        concurrency = $Concurrency
        warmup_requests = $WarmupRequests
        repetitions = $Repetitions
        requests_per_point = $RequestsPerPoint
    }
    profiling = [ordered]@{
        input_tokens = $ProfileInputTokens
        output_tokens = $ProfileOutputTokens
        num_requests = $ProfileNumRequests
        concurrency = $ProfileConcurrency
    }
    resource = [ordered]@{ max_wait_seconds = $MaxWaitSeconds }
    execution = [ordered]@{
        execution_mode = $ExecutionMode
        allow_service_mutation = $true
    }
}
if (-not [string]::IsNullOrWhiteSpace($SpecFile)) {
    if (-not (Test-Path -LiteralPath $SpecFile -PathType Leaf)) {
        throw "SpecFile not found: $SpecFile"
    }
    $spec = Get-Content -LiteralPath $SpecFile -Raw | ConvertFrom-Json
}

if ($DryRun) {
    Write-Host "DRY RUN - no server changes."
    Write-Host "RunId: $RunId"
    Write-Host "Remote run: $remoteRun"
    $spec | ConvertTo-Json -Depth 8
    exit 0
}

$temporarySpec = Join-Path ([IO.Path]::GetTempPath()) "$RunId-spec.json"
try {
    $specJson = $spec | ConvertTo-Json -Depth 12
    [IO.File]::WriteAllText(
        $temporarySpec,
        $specJson,
        [Text.UTF8Encoding]::new($false)
    )
    Write-Host "[1/3] Preparing remote run and updating the local test tool copy ..."
    Invoke-Ssh "mkdir -p '$remoteTool' '$remoteRun'"
    $upload = @($requiredFiles | ForEach-Object { Join-Path $toolDir $_ })
    & scp -P $SshPort @upload "${target}:$remoteTool/"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to upload benchmark modules."
    }
    & scp -P $SshPort $selectorScript "${target}:$RemoteProject/find_idle_npu.sh"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to upload find_idle_npu.sh."
    }
    Invoke-Ssh "chmod +x '$RemoteProject/find_idle_npu.sh'"
    & scp -P $SshPort $temporarySpec "${target}:$remoteSpec"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to upload spec.json."
    }
    Write-Host "[2/3] Launching the controller with nohup ..."
    $launch = "cd '$remoteTool' && " +
        "nohup python3 '$remoteController' --run-dir '$remoteRun' --spec '$remoteSpec' " +
        "> '$remoteRun/controller.log' 2>&1 < /dev/null & echo `$! > '$remoteRun/controller.pid'"
    Invoke-Ssh $launch
    Write-Host "[3/3] Submitted. SSH can now disconnect without stopping the task."
    Write-Host "RunId: $RunId"
    Write-Host "Status: .\parallel_bench.ps1 -Action Status -RunId $RunId"
    Write-Host "Fetch:  .\parallel_bench.ps1 -Action Fetch -RunId $RunId"
} finally {
    Remove-Item -LiteralPath $temporarySpec -Force -ErrorAction SilentlyContinue
}
