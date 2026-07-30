# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

[CmdletBinding()]
param(
    [ValidateSet("Submit", "Status", "Fetch", "Resume", "Cancel")]
    [string]$Action = "Submit",
    [string]$ConfigFile = (Join-Path $PSScriptRoot "configs\parallel_bench_config.json"),
    [string]$SpecFile = "",
    [string]$RunId = "",
    [string]$Server = "192.168.13.190",
    [string]$SshUser = "root",
    [ValidateRange(1, 65535)]
    [int]$SshPort = 22,
    [string]$OutputDir = "D:\vllm-parallel-bench",
    [string]$RemoteProject = "/home/vllm/l00977701/pipeline_parallel",
    [string]$RemoteRunRoot = "/home/vllm/l00977701/runtime/parallel_bench_runs",
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
    [ValidateRange(1, 21600)]
    [int]$MaxWaitSeconds = 21600,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$script:CliParameters = @{} + $PSBoundParameters
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

function Test-ExplicitParameter {
    param([Parameter(Mandatory = $true)][string]$Name)
    return $script:CliParameters.ContainsKey($Name)
}

function Get-OptionalProperty {
    param(
        [object]$Object,
        [Parameter(Mandatory = $true)][string]$Name,
        [object]$Default = $null
    )
    if ($null -ne $Object -and
        $null -ne $Object.PSObject.Properties[$Name]) {
        return $Object.$Name
    }
    return $Default
}

$selectedConfig = if (-not [string]::IsNullOrWhiteSpace($SpecFile)) {
    $SpecFile
} else {
    $ConfigFile
}
if (-not (Test-Path -LiteralPath $selectedConfig -PathType Leaf)) {
    throw "ConfigFile not found: $selectedConfig"
}
$configPath = (Resolve-Path -LiteralPath $selectedConfig).Path
$config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
$client = Get-OptionalProperty $config "client"

if ($null -ne $client) {
    if (-not (Test-ExplicitParameter "Server")) {
        $Server = [string](Get-OptionalProperty $client "server" $Server)
    }
    if (-not (Test-ExplicitParameter "SshUser")) {
        $SshUser = [string](Get-OptionalProperty $client "ssh_user" $SshUser)
    }
    if (-not (Test-ExplicitParameter "SshPort")) {
        $SshPort = [int](Get-OptionalProperty $client "ssh_port" $SshPort)
    }
    if (-not (Test-ExplicitParameter "OutputDir")) {
        $OutputDir = [string](Get-OptionalProperty $client "output_dir" $OutputDir)
    }
    if (-not (Test-ExplicitParameter "RemoteProject")) {
        $RemoteProject = [string](
            Get-OptionalProperty $client "remote_project" $RemoteProject
        )
    }
    if (-not (Test-ExplicitParameter "RemoteRunRoot")) {
        $RemoteRunRoot = [string](
            Get-OptionalProperty $client "remote_run_root" $RemoteRunRoot
        )
    }
}

$containerConfig = Get-OptionalProperty $config "container"
$modelConfig = Get-OptionalProperty $config "model"
$matrixConfig = Get-OptionalProperty $config "matrix"
$workloadConfig = Get-OptionalProperty $config "workload"
$profileConfig = Get-OptionalProperty $config "profiling"
$resourceConfig = Get-OptionalProperty $config "resource"
$executionConfig = Get-OptionalProperty $config "execution"

if ($null -ne $containerConfig) {
    if (-not (Test-ExplicitParameter "Container")) {
        $Container = [string](Get-OptionalProperty $containerConfig "name" $Container)
    }
    if (-not (Test-ExplicitParameter "ExpectedVllmVersion")) {
        $ExpectedVllmVersion = [string](
            Get-OptionalProperty $containerConfig "expected_vllm_version" ""
        )
    }
    if (-not (Test-ExplicitParameter "ExpectedVllmAscendVersion")) {
        $ExpectedVllmAscendVersion = [string](
            Get-OptionalProperty $containerConfig "expected_vllm_ascend_version" ""
        )
    }
}
if ($null -ne $modelConfig) {
    if (-not (Test-ExplicitParameter "Model")) {
        $Model = [string](Get-OptionalProperty $modelConfig "path" $Model)
    }
    if (-not (Test-ExplicitParameter "ModelKind")) {
        $ModelKind = [string](Get-OptionalProperty $modelConfig "kind" $ModelKind)
    }
}
if ($null -ne $matrixConfig -and -not (Test-ExplicitParameter "Matrix")) {
    $Matrix = [string](Get-OptionalProperty $matrixConfig "preset" $Matrix)
}
if ($null -ne $executionConfig -and
    -not (Test-ExplicitParameter "ExecutionMode")) {
    $ExecutionMode = [string](
        Get-OptionalProperty $executionConfig "execution_mode" $ExecutionMode
    )
}
if ($null -ne $workloadConfig) {
    if (-not (Test-ExplicitParameter "InputTokens")) {
        $InputTokens = @($workloadConfig.input_tokens)
    }
    if (-not (Test-ExplicitParameter "OutputTokens")) {
        $OutputTokens = @($workloadConfig.output_tokens)
    }
    if (-not (Test-ExplicitParameter "Concurrency")) {
        $Concurrency = @($workloadConfig.concurrency)
    }
    if (-not (Test-ExplicitParameter "RequestsPerPoint")) {
        $RequestsPerPoint = [int](
            Get-OptionalProperty $workloadConfig "requests_per_point" $RequestsPerPoint
        )
    }
    if (-not (Test-ExplicitParameter "Repetitions")) {
        $Repetitions = [int](
            Get-OptionalProperty $workloadConfig "repetitions" $Repetitions
        )
    }
    if (-not (Test-ExplicitParameter "WarmupRequests")) {
        $WarmupRequests = [int](
            Get-OptionalProperty $workloadConfig "warmup_requests" $WarmupRequests
        )
    }
}
if ($null -ne $profileConfig) {
    if (-not (Test-ExplicitParameter "ProfileNumRequests")) {
        $ProfileNumRequests = [int](
            Get-OptionalProperty $profileConfig "num_requests" $ProfileNumRequests
        )
    }
    if (-not (Test-ExplicitParameter "ProfileConcurrency")) {
        $ProfileConcurrency = [int](
            Get-OptionalProperty $profileConfig "concurrency" $ProfileConcurrency
        )
    }
    if (-not (Test-ExplicitParameter "ProfileInputTokens")) {
        $ProfileInputTokens = [int](
            Get-OptionalProperty $profileConfig "input_tokens" $ProfileInputTokens
        )
    }
    if (-not (Test-ExplicitParameter "ProfileOutputTokens")) {
        $ProfileOutputTokens = [int](
            Get-OptionalProperty $profileConfig "output_tokens" $ProfileOutputTokens
        )
    }
}
if ($null -ne $resourceConfig -and
    -not (Test-ExplicitParameter "MaxWaitSeconds")) {
    $MaxWaitSeconds = [int](
        Get-OptionalProperty $resourceConfig "max_wait_seconds" $MaxWaitSeconds
    )
}

function Resolve-RunId {
    $configuredRun = Get-OptionalProperty (
        Get-OptionalProperty $config "run"
    ) "run_id" ""
    if (-not [string]::IsNullOrWhiteSpace($RunId)) {
        return $RunId
    }
    if (-not [string]::IsNullOrWhiteSpace([string]$configuredRun) -and
        [string]$configuredRun -ne "auto") {
        return [string]$configuredRun
    }
    if ($Action -eq "Submit") {
        return "parallel-" + (Get-Date -Format "yyyyMMdd-HHmmss")
    }
    $lastRunPath = "$configPath.last-run.json"
    if (Test-Path -LiteralPath $lastRunPath -PathType Leaf) {
        $lastRun = Get-Content -LiteralPath $lastRunPath -Raw | ConvertFrom-Json
        if (-not [string]::IsNullOrWhiteSpace([string]$lastRun.run_id)) {
            return [string]$lastRun.run_id
        }
    }
    throw "No RunId is available. Submit once with this config or set run.run_id."
}

$RunId = Resolve-RunId
if ($RunId -notmatch "^[A-Za-z0-9][A-Za-z0-9_.-]*$") {
    throw "RunId contains unsafe characters."
}
if ($SshPort -lt 1 -or $SshPort -gt 65535) {
    throw "client.ssh_port must be between 1 and 65535."
}

$target = "${SshUser}@${Server}"
$remoteTool = "$RemoteProject/profile_tool"
$remoteRun = "$RemoteRunRoot/$RunId"
$remoteSpec = "$remoteRun/spec.json"
$remoteController = "$remoteTool/benchmark_remote.py"
$persistSeconds = [int](Get-OptionalProperty $client "ssh_control_persist_seconds" 600)
if ($persistSeconds -lt 30 -or $persistSeconds -gt 86400) {
    throw "client.ssh_control_persist_seconds must be between 30 and 86400."
}
$multiplexing = [bool](Get-OptionalProperty $client "ssh_multiplexing" $true)

$spec = $config | ConvertTo-Json -Depth 20 | ConvertFrom-Json
$spec.PSObject.Properties.Remove("client")
$spec.PSObject.Properties.Remove("run")
if ($null -eq $spec.PSObject.Properties["server"]) {
    $spec | Add-Member -NotePropertyName "server" -NotePropertyValue ([pscustomobject]@{})
}
$spec.server | Add-Member -Force -NotePropertyName "host" -NotePropertyValue $Server
$spec.server | Add-Member -Force -NotePropertyName "ssh_user" -NotePropertyValue $SshUser
$spec.server | Add-Member -Force -NotePropertyName "remote_root" -NotePropertyValue $RemoteRunRoot
$spec.container | Add-Member -Force -NotePropertyName "name" -NotePropertyValue $Container
$spec.container | Add-Member -Force -NotePropertyName "expected_vllm_version" `
    -NotePropertyValue $ExpectedVllmVersion
$spec.container | Add-Member -Force -NotePropertyName "expected_vllm_ascend_version" `
    -NotePropertyValue $ExpectedVllmAscendVersion
$spec.model | Add-Member -Force -NotePropertyName "path" -NotePropertyValue $Model
$spec.model | Add-Member -Force -NotePropertyName "kind" -NotePropertyValue $ModelKind
$spec.matrix | Add-Member -Force -NotePropertyName "preset" -NotePropertyValue $Matrix
$spec.workload | Add-Member -Force -NotePropertyName "input_tokens" `
    -NotePropertyValue @($InputTokens)
$spec.workload | Add-Member -Force -NotePropertyName "output_tokens" `
    -NotePropertyValue @($OutputTokens)
$spec.workload | Add-Member -Force -NotePropertyName "concurrency" `
    -NotePropertyValue @($Concurrency)
$spec.workload | Add-Member -Force -NotePropertyName "warmup_requests" `
    -NotePropertyValue $WarmupRequests
$spec.workload | Add-Member -Force -NotePropertyName "repetitions" `
    -NotePropertyValue $Repetitions
$spec.workload | Add-Member -Force -NotePropertyName "requests_per_point" `
    -NotePropertyValue $RequestsPerPoint
$spec.profiling | Add-Member -Force -NotePropertyName "input_tokens" `
    -NotePropertyValue $ProfileInputTokens
$spec.profiling | Add-Member -Force -NotePropertyName "output_tokens" `
    -NotePropertyValue $ProfileOutputTokens
$spec.profiling | Add-Member -Force -NotePropertyName "num_requests" `
    -NotePropertyValue $ProfileNumRequests
$spec.profiling | Add-Member -Force -NotePropertyName "concurrency" `
    -NotePropertyValue $ProfileConcurrency
$spec.execution | Add-Member -Force -NotePropertyName "execution_mode" `
    -NotePropertyValue $ExecutionMode
$spec.resource | Add-Member -Force -NotePropertyName "max_wait_seconds" `
    -NotePropertyValue $MaxWaitSeconds

Write-Host "Config: $configPath"
Write-Host "RunId: $RunId"
Write-Host "Target: $target"
Write-Host "Remote run: $remoteRun"
Write-Host "Windows output: $(Join-Path $OutputDir $RunId)"

if ($DryRun) {
    Write-Host "DRY RUN - no server changes."
    $spec | ConvertTo-Json -Depth 12
    return
}

foreach ($program in @("ssh", "scp")) {
    if (-not (Get-Command $program -ErrorAction SilentlyContinue)) {
        throw "$program was not found. Install Windows OpenSSH Client."
    }
}

$sshArgs = @("-p", "$SshPort")
$scpArgs = @("-P", "$SshPort")
if ($multiplexing) {
    $identity = [Text.Encoding]::UTF8.GetBytes("$target-$SshPort-$PID")
    $hasher = [Security.Cryptography.SHA256]::Create()
    try {
        $hash = $hasher.ComputeHash($identity)
    } finally {
        $hasher.Dispose()
    }
    $digest = ([BitConverter]::ToString($hash) -replace "-", "").Substring(
        0, 16
    ).ToLowerInvariant()
    $controlPath = (
        Join-Path ([IO.Path]::GetTempPath()) "vpb-$digest.sock"
    ).Replace("\", "/")
    $controlOptions = @(
        "-o", "ControlMaster=auto",
        "-o", "ControlPersist=${persistSeconds}s",
        "-o", "ControlPath=$controlPath"
    )
    $sshArgs += $controlOptions
    $scpArgs += $controlOptions
}

function Open-SshMaster {
    if (-not $multiplexing) {
        return
    }
    Write-Host "Opening one reusable SSH connection (password is requested once) ..."
    & ssh @sshArgs $target "true"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to open the reusable SSH connection."
    }
}

function Close-SshMaster {
    if (-not $multiplexing) {
        return
    }
    & ssh @sshArgs -O exit $target 2>$null
}

function Invoke-Ssh {
    param([Parameter(Mandatory = $true)][string]$Command)
    & ssh @sshArgs $target $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Remote command failed with exit code $LASTEXITCODE."
    }
}

function Invoke-Scp {
    param([Parameter(Mandatory = $true)][object[]]$Arguments)
    & scp @scpArgs @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "SCP failed with exit code $LASTEXITCODE."
    }
}

Open-SshMaster
try {
    switch ($Action) {
        "Status" {
            $statusCommand = "test -s '$remoteRun/state.json' && cat '$remoteRun/state.json' " +
                "|| { echo 'state not found'; exit 2; }; " +
                "pid=''; test -s '$remoteRun/controller.pid' && pid=`$(cat '$remoteRun/controller.pid'); " +
                "case `"`$pid`" in (*[!0-9]*|`"`") echo 'CONTROLLER_ALIVE=false';; " +
                "(*) if kill -0 `"`$pid`" 2>/dev/null; then echo 'CONTROLLER_ALIVE=true'; " +
                "else echo 'CONTROLLER_ALIVE=false'; fi;; esac"
            Invoke-Ssh $statusCommand
            return
        }
        "Fetch" {
            New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
            Invoke-Scp -Arguments @("-r", "${target}:$remoteRun", $OutputDir)
            Write-Host "Downloaded: $(Join-Path $OutputDir $RunId)"
            Write-Host "Report: $(Join-Path (Join-Path $OutputDir $RunId) 'report.html')"
            return
        }
        "Cancel" {
            Invoke-Ssh "touch '$remoteRun/CANCEL'"
            Write-Host "Cancellation requested."
            return
        }
        "Resume" {
            $launch = "cd '$remoteTool' && test -s '$remoteSpec' && " +
                "rm -f '$remoteRun/CANCEL' && " +
                "nohup python3 '$remoteController' --run-dir '$remoteRun' --spec '$remoteSpec' " +
                ">> '$remoteRun/controller.log' 2>&1 < /dev/null & echo `$! > '$remoteRun/controller.pid'"
            Invoke-Ssh $launch
            Write-Host "Resumed detached run: $RunId"
            return
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

    $temporarySpec = Join-Path ([IO.Path]::GetTempPath()) "$RunId-spec.json"
    try {
        [IO.File]::WriteAllText(
            $temporarySpec,
            ($spec | ConvertTo-Json -Depth 20),
            [Text.UTF8Encoding]::new($false)
        )
        Write-Host "[1/3] Preparing the remote run and uploading tools ..."
        Invoke-Ssh "mkdir -p '$remoteTool' '$remoteRun'"
        $upload = @($requiredFiles | ForEach-Object { Join-Path $toolDir $_ })
        Invoke-Scp -Arguments @($upload + "${target}:$remoteTool/")
        Invoke-Scp -Arguments @(
            $selectorScript,
            "${target}:$RemoteProject/find_idle_npu.sh"
        )
        Invoke-Ssh "chmod +x '$RemoteProject/find_idle_npu.sh'"
        Invoke-Scp -Arguments @($temporarySpec, "${target}:$remoteSpec")

        Write-Host "[2/3] Launching the controller with nohup ..."
        $launch = "cd '$remoteTool' && " +
            "nohup python3 '$remoteController' --run-dir '$remoteRun' --spec '$remoteSpec' " +
            "> '$remoteRun/controller.log' 2>&1 < /dev/null & echo `$! > '$remoteRun/controller.pid'"
        Invoke-Ssh $launch

        $lastRun = [ordered]@{
            run_id = $RunId
            submitted_at = (Get-Date).ToString("o")
            server = $Server
            remote_run = $remoteRun
            output_dir = $OutputDir
        }
        [IO.File]::WriteAllText(
            "$configPath.last-run.json",
            ($lastRun | ConvertTo-Json),
            [Text.UTF8Encoding]::new($false)
        )
        Write-Host "[3/3] Submitted. SSH can disconnect without stopping the task."
        Write-Host "Status: .\parallel_bench.ps1 -Action Status -ConfigFile `"$configPath`""
        Write-Host "Fetch:  .\parallel_bench.ps1 -Action Fetch -ConfigFile `"$configPath`""
    } finally {
        Remove-Item -LiteralPath $temporarySpec -Force -ErrorAction SilentlyContinue
    }
} finally {
    Close-SshMaster
}
