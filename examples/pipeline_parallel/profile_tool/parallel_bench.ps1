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
    [string]$Image = "quay.io/ascend/vllm-ascend:nightly-main-a3",
    [string]$Model = "/home/vllm/l00977701/models/Qwen3-30B-A3B",
    [ValidateSet("moe", "dense", "auto")]
    [string]$ModelKind = "moe",
    [ValidateSet("quick", "boundary", "custom")]
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
    [ValidateRange(1, 300)]
    [int]$SshConnectTimeoutSeconds = 15,
    [ValidateRange(5, 300)]
    [int]$SshServerAliveIntervalSeconds = 15,
    [ValidateRange(1, 10)]
    [int]$SshServerAliveCountMax = 3,
    [ValidateRange(30, 3600)]
    [int]$RemoteCommandTimeoutSeconds = 600,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$script:CliParameters = @{} + $PSBoundParameters
$toolDir = $PSScriptRoot
$dockerScript = Join-Path (Split-Path $PSScriptRoot -Parent) "docker.sh"
$selectorScript = Join-Path (Split-Path $PSScriptRoot -Parent) "find_idle_npu.sh"
$requiredFiles = @(
    "benchmark_remote.py",
    "experiment_schema.py",
    "resource_scheduler.py",
    "workload_streaming.py",
    "profiler_collect.py",
    "metrics_analyzer.py",
    "report_generator.py",
    "install_profile_scopes.py"
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
    if (-not (Test-ExplicitParameter "SshConnectTimeoutSeconds")) {
        $SshConnectTimeoutSeconds = [int](
            Get-OptionalProperty $client "ssh_connect_timeout_seconds" `
                $SshConnectTimeoutSeconds
        )
    }
    if (-not (Test-ExplicitParameter "SshServerAliveIntervalSeconds")) {
        $SshServerAliveIntervalSeconds = [int](
            Get-OptionalProperty $client "ssh_server_alive_interval_seconds" `
                $SshServerAliveIntervalSeconds
        )
    }
    if (-not (Test-ExplicitParameter "SshServerAliveCountMax")) {
        $SshServerAliveCountMax = [int](
            Get-OptionalProperty $client "ssh_server_alive_count_max" `
                $SshServerAliveCountMax
        )
    }
    if (-not (Test-ExplicitParameter "RemoteCommandTimeoutSeconds")) {
        $RemoteCommandTimeoutSeconds = [int](
            Get-OptionalProperty $client "remote_command_timeout_seconds" `
                $RemoteCommandTimeoutSeconds
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
    if (-not (Test-ExplicitParameter "Image")) {
        $Image = [string](Get-OptionalProperty $containerConfig "image" $Image)
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
if ($Model -notmatch "^/[A-Za-z0-9._/-]+$") {
    throw "model.path must be a safe absolute host path without spaces."
}
if ($Container -notmatch "^[A-Za-z0-9][A-Za-z0-9_.-]*$") {
    throw "container.name contains unsafe characters."
}
if ($Image -notmatch "^[A-Za-z0-9][A-Za-z0-9._/:@-]*$") {
    throw "container.image contains unsafe characters."
}

$target = "${SshUser}@${Server}"
$remoteTool = "$RemoteProject/profile_tool"
$remoteRun = "$RemoteRunRoot/$RunId"
$remoteSpec = "$remoteRun/spec.json"
$remoteController = "$remoteTool/benchmark_remote.py"
$remoteDocker = "$RemoteProject/docker.sh"
$persistSeconds = [int](Get-OptionalProperty $client "ssh_control_persist_seconds" 600)
if ($persistSeconds -lt 30 -or $persistSeconds -gt 86400) {
    throw "client.ssh_control_persist_seconds must be between 30 and 86400."
}
$multiplexing = [bool](Get-OptionalProperty $client "ssh_multiplexing" $true)
if ($multiplexing -and $env:OS -eq "Windows_NT") {
    Write-Warning (
        "SSH multiplexing is not supported by Win32 OpenSSH; " +
        "using the batched Windows submit workflow instead."
    )
    $multiplexing = $false
}

$spec = $config | ConvertTo-Json -Depth 20 | ConvertFrom-Json
$spec.PSObject.Properties.Remove("client")
$spec.PSObject.Properties.Remove("run")
if ($null -eq $spec.PSObject.Properties["server"]) {
    $spec | Add-Member -NotePropertyName "server" -NotePropertyValue ([pscustomobject]@{})
}
if ($null -eq $spec.PSObject.Properties["profiling"]) {
    $spec | Add-Member -NotePropertyName "profiling" -NotePropertyValue ([pscustomobject]@{})
}
$spec.server | Add-Member -Force -NotePropertyName "host" -NotePropertyValue $Server
$spec.server | Add-Member -Force -NotePropertyName "ssh_user" -NotePropertyValue $SshUser
$spec.server | Add-Member -Force -NotePropertyName "remote_root" -NotePropertyValue $RemoteRunRoot
$spec.container | Add-Member -Force -NotePropertyName "name" -NotePropertyValue $Container
$spec.container | Add-Member -Force -NotePropertyName "image" -NotePropertyValue $Image
$spec.container | Add-Member -Force -NotePropertyName "expected_vllm_version" `
    -NotePropertyValue $ExpectedVllmVersion
$spec.container | Add-Member -Force -NotePropertyName "expected_vllm_ascend_version" `
    -NotePropertyValue $ExpectedVllmAscendVersion
$spec.model | Add-Member -Force -NotePropertyName "path" -NotePropertyValue $Model
$spec.model | Add-Member -Force -NotePropertyName "container_path" -NotePropertyValue "/models"
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

$requiredPrograms = @("ssh")
if ($Action -in @("Submit", "Fetch")) {
    $requiredPrograms += "scp"
}
if ($Action -in @("Submit", "Fetch")) {
    $requiredPrograms += "tar"
}
foreach ($program in $requiredPrograms) {
    if (-not (Get-Command $program -ErrorAction SilentlyContinue)) {
        throw "$program was not found. Install Windows OpenSSH Client."
    }
}

$connectionOptions = @(
    "-o", "ConnectTimeout=$SshConnectTimeoutSeconds",
    "-o", "ConnectionAttempts=1",
    "-o", "ServerAliveInterval=$SshServerAliveIntervalSeconds",
    "-o", "ServerAliveCountMax=$SshServerAliveCountMax"
)
$sshArgs = @("-p", "$SshPort") + $connectionOptions
# OpenSSH 9+ uses SFTP for scp by default. Some benchmark hosts expose an
# interactive SSH shell but do not configure the SFTP subsystem, which closes
# the upload immediately with exit code 255. The payload is a regular tar file,
# so the legacy SCP wire protocol is sufficient and more widely compatible.
$scpArgs = @("-O", "-P", "$SshPort") + $connectionOptions
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
    param(
        [Parameter(Mandatory = $true)][string]$Command,
        [string]$Operation = "remote command",
        [int]$TimeoutSeconds = $RemoteCommandTimeoutSeconds
    )
    $commandBytes = [Text.Encoding]::UTF8.GetBytes($Command)
    $encodedCommand = [Convert]::ToBase64String($commandBytes)
    $remoteCommand = (
        "printf %s $encodedCommand | base64 -d | " +
        "timeout --foreground ${TimeoutSeconds}s bash"
    )
    Write-Host "  -> $Operation (timeout: ${TimeoutSeconds}s)"
    & ssh @sshArgs $target $remoteCommand
    $exitCode = $LASTEXITCODE
    if ($exitCode -eq 124) {
        throw "$Operation timed out after $TimeoutSeconds seconds."
    }
    if ($exitCode -ne 0) {
        throw "$Operation failed with exit code $exitCode."
    }
}

function Invoke-Scp {
    param(
        [Parameter(Mandatory = $true)][object[]]$Arguments,
        [string]$Operation = "file upload"
    )
    Write-Host "  -> $Operation"
    & scp @scpArgs @Arguments
    if ($LASTEXITCODE -ne 0) {
        if ($LASTEXITCODE -eq 255) {
            Write-Host (
                "SCP connection closed. The script already forced the " +
                "legacy SCP protocol; verify the SSH password, sshd " +
                "MaxSessions/AllowUsers policy, and available space in /tmp."
            )
        }
        throw "$Operation failed with exit code $LASTEXITCODE."
    }
}

function Write-LfCopy {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Destination
    )
    $content = [IO.File]::ReadAllText($Source)
    $normalized = $content.Replace("`r`n", "`n").Replace("`r", "`n")
    [IO.File]::WriteAllText(
        $Destination,
        $normalized,
        [Text.UTF8Encoding]::new($false)
    )
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
            Invoke-Ssh $statusCommand -Operation "reading remote status"
            return
        }
        "Fetch" {
            New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
            $remoteFetchArchive = "/tmp/$RunId-fetch.tar.gz"
            $remoteFetchList = "/tmp/$RunId-fetch-files"
            $localFetchArchive = Join-Path (
                [IO.Path]::GetTempPath()
            ) "$RunId-fetch.tar.gz"
            $fetchCommand = "set -e; cd '$RemoteRunRoot'; " +
                "test -d '$RunId'; rm -f '$remoteFetchArchive' '$remoteFetchList'; " +
                "find '$RunId' -path '$RunId/profiles' -prune -o -type f -print0 " +
                "> '$remoteFetchList'; " +
                "if test -d '$RunId/profiles'; then " +
                "find '$RunId/profiles' -type f -name 'merged_trace_view.json' " +
                "-print0 >> '$remoteFetchList'; fi; " +
                "tar --null --files-from='$remoteFetchList' " +
                "-czf '$remoteFetchArchive'; rm -f '$remoteFetchList'; " +
                "nohup sh -c 'sleep 3600; rm -f $remoteFetchArchive' " +
                ">/dev/null 2>&1 &"
            try {
                Write-Host "[1/3] Packaging filtered results on the remote host ..."
                Invoke-Ssh $fetchCommand -Operation "packaging filtered run results"
                Write-Host "[2/3] Downloading one result archive ..."
                Invoke-Scp -Arguments @(
                    "${target}:$remoteFetchArchive",
                    $localFetchArchive
                ) -Operation "downloading filtered run results"
                Write-Host "[3/3] Extracting results locally ..."
                & tar -xzf $localFetchArchive -C $OutputDir
                if ($LASTEXITCODE -ne 0) {
                    throw "Failed to extract the downloaded result archive."
                }
            } finally {
                Remove-Item -LiteralPath $localFetchArchive -Force `
                    -ErrorAction SilentlyContinue
            }
            Write-Host "Downloaded: $(Join-Path $OutputDir $RunId)"
            Write-Host "Profiles: merged_trace_view.json only"
            Write-Host "Report: $(Join-Path (Join-Path $OutputDir $RunId) 'report.html')"
            return
        }
        "Cancel" {
            Invoke-Ssh "touch '$remoteRun/CANCEL'" `
                -Operation "requesting cancellation"
            Write-Host "Cancellation requested."
            return
        }
        "Resume" {
            $launch = "cd '$remoteTool' && test -s '$remoteSpec' && " +
                "rm -f '$remoteRun/CANCEL' && " +
                "nohup python3 '$remoteController' --run-dir '$remoteRun' --spec '$remoteSpec' " +
                ">> '$remoteRun/controller.log' 2>&1 < /dev/null & echo `$! > '$remoteRun/controller.pid'"
            Invoke-Ssh $launch -Operation "resuming detached controller"
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
    if (-not (Test-Path -LiteralPath $dockerScript -PathType Leaf)) {
        throw "Missing bundled Docker launcher: $dockerScript"
    }

    $temporarySpec = Join-Path ([IO.Path]::GetTempPath()) "$RunId-spec.json"
    $temporaryDocker = Join-Path ([IO.Path]::GetTempPath()) "$RunId-docker.sh"
    $temporarySelector = Join-Path ([IO.Path]::GetTempPath()) "$RunId-find-idle.sh"
    $temporaryBundle = Join-Path ([IO.Path]::GetTempPath()) "$RunId-submit"
    $temporaryArchive = Join-Path ([IO.Path]::GetTempPath()) "$RunId-submit.tar"
    try {
        Write-LfCopy -Source $dockerScript -Destination $temporaryDocker
        Write-LfCopy -Source $selectorScript -Destination $temporarySelector
        [IO.File]::WriteAllText(
            $temporarySpec,
            ($spec | ConvertTo-Json -Depth 20),
            [Text.UTF8Encoding]::new($false)
        )
        New-Item -ItemType Directory -Force -Path (
            Join-Path $temporaryBundle "profile_tool"
        ) | Out-Null
        foreach ($file in $requiredFiles) {
            Copy-Item -LiteralPath (Join-Path $toolDir $file) -Destination (
                Join-Path (Join-Path $temporaryBundle "profile_tool") $file
            )
        }
        Copy-Item -LiteralPath $temporarySelector -Destination (
            Join-Path $temporaryBundle "find_idle_npu.sh"
        )
        Copy-Item -LiteralPath $temporaryDocker -Destination (
            Join-Path $temporaryBundle "docker.sh"
        )
        Copy-Item -LiteralPath $temporarySpec -Destination (
            Join-Path $temporaryBundle "spec.json"
        )
        & tar -cf $temporaryArchive -C $temporaryBundle .
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to create the submit bundle."
        }

        $remoteArchive = "/tmp/$RunId-submit.tar"
        $remoteUpload = "$remoteRun/.submit-upload"
        $profilePolicy = [string](
            Get-OptionalProperty $spec.execution "profile_policy" "representative_and_boundary"
        )
        $explicitCases = @($spec.matrix.cases)
        $hasProfileCase = @(
            $explicitCases | Where-Object { [bool]$_.profile }
        ).Count -gt 0
        $installProfileScopes = (
            $profilePolicy -ne "none" -and
            ($explicitCases.Count -eq 0 -or $hasProfileCase)
        )
        $profileInstallCommand = if ($installProfileScopes) {
            "echo '[remote 2/3] Installing high-level profiling scopes'; " +
            "docker exec '$Container' python3 " +
            "'/workspace/pipeline_parallel/profile_tool/install_profile_scopes.py'; "
        } else {
            "echo '[remote 2/3] Profiling disabled; skipping scope installation'; "
        }
        Write-Host "[1/3] Uploading one bundled payload (password prompt 1/2) ..."
        Invoke-Scp -Arguments @($temporaryArchive, "${target}:$remoteArchive") `
            -Operation "uploading bundled benchmark payload"

        Write-Host "[2/3] Preparing container and controller (password prompt 2/2) ..."
        $submitCommand = "set -e; " +
            "echo '[remote 1/3] Installing uploaded tools'; " +
            "rm -rf '$remoteUpload'; mkdir -p '$remoteUpload' '$remoteTool' '$remoteRun'; " +
            "tar -xf '$remoteArchive' -C '$remoteUpload'; rm -f '$remoteArchive'; " +
            "cp -f '$remoteUpload/profile_tool/'*.py '$remoteTool/'; " +
            "cp -f '$remoteUpload/find_idle_npu.sh' '$RemoteProject/find_idle_npu.sh'; " +
            "cp -f '$remoteUpload/docker.sh' '$remoteDocker'; " +
            "cp -f '$remoteUpload/spec.json' '$remoteSpec'; rm -rf '$remoteUpload'; " +
            "chmod +x '$RemoteProject/find_idle_npu.sh' '$remoteDocker'; " +
            "echo '[remote 2/3] Recreating the test container'; " +
            "IMAGE='$Image' CONTAINER_NAME='$Container' MODEL_DIR='$Model' " +
            "'$remoteDocker' restart; " +
            $profileInstallCommand +
            "echo '[remote 3/3] Launching the detached controller'; " +
            "cd '$remoteTool'; " +
            "nohup python3 '$remoteController' --run-dir '$remoteRun' --spec '$remoteSpec' " +
            "> '$remoteRun/controller.log' 2>&1 < /dev/null & " +
            "controller_pid=`$!; echo `"`$controller_pid`" > '$remoteRun/controller.pid'"
        Invoke-Ssh $submitCommand -Operation "running remote submit workflow"
        Write-Host "[3/3] Submission complete."

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
        Write-Host "Submitted. SSH can disconnect without stopping the task."
        Write-Host "Status: .\parallel_bench.ps1 -Action Status -ConfigFile `"$configPath`""
        Write-Host "Fetch:  .\parallel_bench.ps1 -Action Fetch -ConfigFile `"$configPath`""
    } finally {
        Remove-Item -LiteralPath @(
            $temporarySpec,
            $temporaryDocker,
            $temporarySelector,
            $temporaryArchive
        ) -Force -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath $temporaryBundle -Recurse -Force `
            -ErrorAction SilentlyContinue
    }
} finally {
    Close-SshMaster
}
