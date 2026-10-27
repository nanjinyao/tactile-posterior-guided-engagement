param(
    [string[]]$Extensions = @(".md", ".xlsx", ".pdf", ".py", ".ps1", ".json", ".txt"),
    [int]$DebounceSeconds = 2
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$updateScript = Join-Path $PSScriptRoot "update_trace.ps1"
$tracePath = Join-Path $repoRoot "project_trace.md"

if (-not (Test-Path -LiteralPath $updateScript)) {
    throw "Update script not found: $updateScript"
}

$watcher = New-Object System.IO.FileSystemWatcher
$watcher.Path = $repoRoot
$watcher.IncludeSubdirectories = $true
$watcher.EnableRaisingEvents = $true

$lastSeen = @{}

function Should-TracePath {
    param([string]$FullPath)

    if ([string]::IsNullOrWhiteSpace($FullPath)) {
        return $false
    }

    $relative = [System.IO.Path]::GetRelativePath($repoRoot, $FullPath)

    if ($relative -like ".git\*" -or $relative -like "node_modules\*") {
        return $false
    }

    if ($FullPath -eq $tracePath) {
        return $false
    }

    $name = [System.IO.Path]::GetFileName($FullPath)
    if ($name -like "~$*") {
        return $false
    }

    $ext = [System.IO.Path]::GetExtension($FullPath)
    return $Extensions -contains $ext
}

function Add-TraceEntry {
    param(
        [string]$ChangeType,
        [string]$FullPath,
        [string]$OldFullPath = ""
    )

    if (-not (Should-TracePath -FullPath $FullPath)) {
        return
    }

    $now = Get-Date
    $key = "$ChangeType|$FullPath"

    if ($lastSeen.ContainsKey($key)) {
        $elapsed = ($now - $lastSeen[$key]).TotalSeconds
        if ($elapsed -lt $DebounceSeconds) {
            return
        }
    }
    $lastSeen[$key] = $now

    $relative = [System.IO.Path]::GetRelativePath($repoRoot, $FullPath)
    $details = "Observed file-system event from watch_trace.ps1."

    if (-not [string]::IsNullOrWhiteSpace($OldFullPath)) {
        $oldRelative = [System.IO.Path]::GetRelativePath($repoRoot, $OldFullPath)
        $details = "Observed rename from $oldRelative to $relative."
    }

    & $updateScript `
        -Summary "File ${ChangeType}: $relative" `
        -Status "auto-observed" `
        -Files $relative `
        -Details $details | Out-Null
}

$subscriptions = @()
$subscriptions += Register-ObjectEvent -InputObject $watcher -EventName Created -Action {
    Add-TraceEntry -ChangeType "created" -FullPath $Event.SourceEventArgs.FullPath
}
$subscriptions += Register-ObjectEvent -InputObject $watcher -EventName Changed -Action {
    Add-TraceEntry -ChangeType "changed" -FullPath $Event.SourceEventArgs.FullPath
}
$subscriptions += Register-ObjectEvent -InputObject $watcher -EventName Deleted -Action {
    Add-TraceEntry -ChangeType "deleted" -FullPath $Event.SourceEventArgs.FullPath
}
$subscriptions += Register-ObjectEvent -InputObject $watcher -EventName Renamed -Action {
    Add-TraceEntry -ChangeType "renamed" -FullPath $Event.SourceEventArgs.FullPath -OldFullPath $Event.SourceEventArgs.OldFullPath
}

Write-Output "Trace watcher running for $repoRoot"
Write-Output "Press Ctrl+C to stop."

try {
    while ($true) {
        Start-Sleep -Seconds 1
    }
}
finally {
    foreach ($subscription in $subscriptions) {
        Unregister-Event -SubscriptionId $subscription.Id -ErrorAction SilentlyContinue
    }
    $watcher.Dispose()
}
