<#
PowerShell launcher for hf2spdx-ai-bom
Adds optional venv detection/creation and dependency installation.

Usage examples:
  ./scripts/ai-bom.ps1 --help
  ./scripts/ai-bom.ps1 gen openai-community/gpt2 -o output/gpt2.spdx3.json --timeout 30
  ./scripts/ai-bom.ps1 enrich hf output/gpt2.spdx3.json -o orig
  ./scripts/ai-bom.ps1 run openai-community/gpt2 --overwrite --add-comment --timeout 30

Launcher-only flags (stripped before invoking CLI):
  -Setup / --setup       Force create .venv and install requirements (non-interactive)
  -NoSetup / --no-setup  Do not prompt for venv creation
  -Python <path>         Use specific python interpreter
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Resolve repository root relative to this script
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot  = Split-Path -Parent $ScriptDir
$CliPath   = Join-Path $RepoRoot 'src' | Join-Path -ChildPath 'cli.py'
$ReqPath   = Join-Path $RepoRoot 'requirements.txt'
$LocalVenv = Join-Path $RepoRoot '.venv'
$LocalVenvPy = Join-Path $LocalVenv 'Scripts' | Join-Path -ChildPath 'python.exe'

if (-not (Test-Path -Path $CliPath)) {
    Write-Error "Cannot find CLI: $CliPath"
    exit 1
}

# Parse launcher-only flags from $args while preserving CLI args
$setup = $false
$noSetup = $false
$pythonOverride = $null
$forward = New-Object System.Collections.Generic.List[string]
for ($i = 0; $i -lt $args.Count; $i++) {
    $a = [string]$args[$i]
    switch -Regex ($a) {
        '^(--setup|-Setup)$' { $setup = $true; continue }
        '^(--no-setup|-NoSetup)$' { $noSetup = $true; continue }
        '^(--python|-Python)$' {
            if ($i + 1 -lt $args.Count) {
                $pythonOverride = [string]$args[$i+1]
                $i++
                continue
            } else {
                Write-Error 'Missing value for -Python/--python'
                exit 2
            }
        }
        default { $forward.Add($a) }
    }
}

function Resolve-Python {
    param([string]$Override)
    if ($Override) { return $Override }
    if ($env:VIRTUAL_ENV) {
        $venvPy = Join-Path $env:VIRTUAL_ENV 'Scripts' | Join-Path -ChildPath 'python.exe'
        if (Test-Path $venvPy) { return $venvPy }
    }
    if (Test-Path $LocalVenvPy) { return $LocalVenvPy }
    $py = Get-Command python -ErrorAction SilentlyContinue
    if ($py) { return 'python' }
    $pylauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pylauncher) { return 'py' }
    throw 'Python interpreter not found on PATH. Please install Python or activate your virtual environment.'
}

function Initialize-VenvAndDeps {
    param([string]$BasePython)
    if (Test-Path $LocalVenvPy) { return $LocalVenvPy }
    # Interactive prompt unless explicitly disabled/enabled
    if (-not $setup -and -not $noSetup) {
        Write-Host "No local venv detected. Create .venv and install requirements now? [Y/n] " -NoNewline
        $ans = Read-Host
        if ($ans -and $ans.Trim().ToLower() -notin @('y','yes','')) {
            return (Resolve-Python -Override $BasePython)
        }
    } elseif ($noSetup) {
        return (Resolve-Python -Override $BasePython)
    }

    Write-Host "[launcher] Creating venv at $LocalVenv ..."
    & $BasePython -m venv $LocalVenv
    if ($LASTEXITCODE -ne 0) { throw "Failed to create venv using $BasePython" }

    $py = $LocalVenvPy
    Write-Host "[launcher] Upgrading pip ..."
    & $py -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw 'pip upgrade failed' }

    if (Test-Path $ReqPath) {
        Write-Host "[launcher] Installing requirements from $ReqPath ..."
        & $py -m pip install -r $ReqPath
        if ($LASTEXITCODE -ne 0) { throw 'pip install -r requirements.txt failed' }
    } else {
        Write-Host "[launcher] No requirements.txt found; skipping dependency install."
    }
    return $py
}

$Python = Resolve-Python -Override $pythonOverride

# If not using a venv python and allowed, offer to create local venv
$usingLocalVenv = (Test-Path $LocalVenvPy) -and ($Python -eq $LocalVenvPy)
if (-not $usingLocalVenv -and -not $env:VIRTUAL_ENV) {
    $Python = Initialize-VenvAndDeps -BasePython $Python
}

if ($forward.Count -eq 0) {
    & $Python $CliPath --help
    exit $LASTEXITCODE
}

& $Python $CliPath @forward
exit $LASTEXITCODE


