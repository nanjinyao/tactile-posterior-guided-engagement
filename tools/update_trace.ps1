param(
    [Parameter(Mandatory = $true)]
    [string]$Summary,

    [string]$Status = "updated",

    [string[]]$Files = @(),

    [string]$Details = ""
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$tracePath = Join-Path $repoRoot "project_trace.md"

if (-not (Test-Path -LiteralPath $tracePath)) {
    throw "Trace file not found: $tracePath"
}

$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss zzz"
$content = Get-Content -Raw -LiteralPath $tracePath
$content = $content -replace "(?m)^Last updated:.*$", "Last updated: $timestamp"
Set-Content -LiteralPath $tracePath -Value $content -Encoding UTF8

if ($Files.Count -gt 0) {
    $filesText = ($Files | ForEach-Object { "- $_" }) -join "`n"
} else {
    $filesText = "- none"
}

if ([string]::IsNullOrWhiteSpace($Details)) {
    $Details = "No additional details provided."
}

$entry = @"

### $timestamp

Status: $Status

Summary: $Summary

Files:

$filesText

Details:

$Details
"@

Add-Content -LiteralPath $tracePath -Value $entry -Encoding UTF8

Write-Output "Trace updated: $tracePath"
