# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$Server,

    [string]$SshUser = "root",
    [ValidateRange(1, 65535)]
    [int]$SshPort = 22,
    [string]$Container = "qwen3_parallel_nightly",
    [string]$Model = "/models/Qwen3-8B",
    [ValidatePattern("^[0-9]+$")]
    [string]$Device = "0",
    [ValidateRange(1, 65535)]
    [int]$Port = 8010,
    [string]$Prompt = [Text.Encoding]::UTF8.GetString(
        [Convert]::FromBase64String(
            "6K+355So5LiJ5Y+l6K+d6K+05piO5rWB5rC057q/5bm26KGM55qE5bel5L2c5Y6f55CG44CC"
        )
    ),
    [ValidateRange(1, 4096)]
    [int]$MaxTokens = 16,
    [ValidateRange(1, 100000)]
    [int]$NumRequests = 1,
    [ValidateRange(1, 100000)]
    [int]$Concurrency = 1,
    [ValidateSet("eager", "aclgraph")]
    [string]$ExecutionMode = "eager",
    [string]$ExpectedVllmVersion = "",
    [ValidateRange(1, 1048576)]
    [int]$MaxModelLen = 4096,
    [ValidateRange(1, [long]::MaxValue)]
    [long]$KvCacheMemoryBytes = 4294967296,
    [ValidateRange(10, 3600)]
    [int]$StartupTimeoutSeconds = 300,
    [ValidateRange(10, 3600)]
    [int]$RequestTimeoutSeconds = 600,
    [string]$OutputDir = "D:\vllm-profiles",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$helper = Join-Path $PSScriptRoot "profile_remote.py"
if (-not (Test-Path -LiteralPath $helper -PathType Leaf)) {
    throw "Missing bundled helper: $helper"
}
if ($Concurrency -gt $NumRequests) {
    throw "Concurrency ($Concurrency) cannot exceed NumRequests ($NumRequests)."
}

$config = [ordered]@{
    container = $Container
    model = $Model
    device = $Device
    port = $Port
    prompt = $Prompt
    max_tokens = $MaxTokens
    num_requests = $NumRequests
    concurrency = $Concurrency
    execution_mode = $ExecutionMode
    expected_vllm_version = $ExpectedVllmVersion
    max_model_len = $MaxModelLen
    kv_cache_memory_bytes = $KvCacheMemoryBytes
    startup_timeout_seconds = $StartupTimeoutSeconds
    request_timeout_seconds = $RequestTimeoutSeconds
}
$configJson = $config | ConvertTo-Json -Compress
$configBytes = [System.Text.Encoding]::UTF8.GetBytes($configJson)
$configB64 = [Convert]::ToBase64String($configBytes)
$target = "${SshUser}@${Server}"
$remoteHelper = ".vllm_profile_tool/profile_remote.py"
$remoteCommand = "python3 $remoteHelper --config-b64 '$configB64'"

if ($DryRun) {
    Write-Host "DRY RUN - no SSH connection will be made."
    Write-Host "Target: $target"
    Write-Host "Remote command: $remoteCommand"
    Write-Host "Local output root: $OutputDir"
    Write-Host "Config:"
    $config | ConvertTo-Json
    exit 0
}

foreach ($program in @("ssh", "scp")) {
    if (-not (Get-Command $program -ErrorAction SilentlyContinue)) {
        throw "$program was not found. Install the Windows OpenSSH Client first."
    }
}

Write-Host "[1/4] Preparing the helper on $target ..."
& ssh -p $SshPort $target "mkdir -p .vllm_profile_tool"
if ($LASTEXITCODE -ne 0) {
    throw "Failed to create the remote helper directory."
}
& scp -P $SshPort $helper "${target}:$remoteHelper"
if ($LASTEXITCODE -ne 0) {
    throw "Failed to upload profile_remote.py."
}

Write-Host "[2/4] Starting the remote profile workflow ..."
$remoteOutput = @(
    & ssh -p $SshPort $target $remoteCommand 2>&1 |
        ForEach-Object {
            Write-Host "$_"
            "$_"
        }
)
if ($LASTEXITCODE -ne 0) {
    throw "Remote profiling failed. Review the ERROR line and service log above."
}

$traceLine = $remoteOutput |
    Where-Object { "$_" -like "TRACE_EXPORT=*" } |
    Select-Object -Last 1
if (-not $traceLine) {
    throw "The remote tool did not return TRACE_EXPORT."
}
$remoteTrace = "$traceLine".Substring("TRACE_EXPORT=".Length)
$remoteRunDir = [System.IO.Path]::GetDirectoryName(
    $remoteTrace.Replace("/", [System.IO.Path]::DirectorySeparatorChar)
).Replace([System.IO.Path]::DirectorySeparatorChar, "/")
$runId = Split-Path $remoteRunDir -Leaf
$localRunDir = Join-Path $OutputDir $runId
New-Item -ItemType Directory -Force -Path $localRunDir | Out-Null

Write-Host "[3/4] Downloading trace_view.json to $localRunDir ..."
$localTrace = Join-Path $localRunDir "trace_view.json"
& scp -P $SshPort `
    "${target}:$remoteTrace" `
    "${target}:$remoteRunDir/run.json" `
    "${target}:$remoteRunDir/response.json" `
    "${target}:$remoteRunDir/service.log" `
    $localRunDir
if ($LASTEXITCODE -ne 0) {
    throw "Failed to download the profiling result files."
}

$localManifest = Join-Path $localRunDir "run.json"
foreach ($resultFile in @(
    $localTrace,
    $localManifest,
    (Join-Path $localRunDir "response.json"),
    (Join-Path $localRunDir "service.log")
)) {
    if (-not (Test-Path -LiteralPath $resultFile -PathType Leaf)) {
        throw "Expected downloaded file is missing: $resultFile"
    }
}

Write-Host "[4/4] Complete."
Write-Host "Trace: $localTrace"
Write-Host "Run metadata: $localManifest"
Write-Host "Request results: $(Join-Path $localRunDir 'response.json')"
Write-Host "Service log: $(Join-Path $localRunDir 'service.log')"
Write-Host "Open trace_view.json with MindStudio Insight."
