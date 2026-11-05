$ErrorActionPreference = 'Stop'

$project = Split-Path $PSScriptRoot -Parent
$venv = Join-Path $project '.venv'

Write-Host 'Setting up VoxCPM in a local Python environment.' -ForegroundColor Cyan

if (-not (Test-Path (Join-Path $venv 'Scripts\python.exe'))) {
    $candidates = @()
    if ($env:VOXCPM_PYTHON) {
        $candidates += @{ Command = $env:VOXCPM_PYTHON; Arguments = @() }
    }
    $candidates += @(
        @{ Command = 'py'; Arguments = @('-3.12') },
        @{ Command = 'py'; Arguments = @('-3.11') },
        @{ Command = 'py'; Arguments = @('-3.10') },
        @{ Command = 'python'; Arguments = @() }
    )

    $created = $false
    foreach ($candidate in $candidates) {
        if (-not (Get-Command $candidate.Command -ErrorAction SilentlyContinue)) {
            continue
        }
        & $candidate.Command @($candidate.Arguments) -c "import sys; sys.exit(not ((3, 10) <= sys.version_info[:2] < (3, 13)))"
        if ($LASTEXITCODE -ne 0) {
            continue
        }
        & $candidate.Command @($candidate.Arguments) -m venv $venv
        if ($LASTEXITCODE -eq 0) {
            $created = $true
            break
        }
    }
    if (-not $created) {
        throw 'Python 3.10, 3.11, or 3.12 is required. Install Python and run this file again.'
    }
}

$python = Join-Path $venv 'Scripts\python.exe'
& $python -c "import sys; sys.exit(not ((3, 10) <= sys.version_info[:2] < (3, 13)))"
if ($LASTEXITCODE -ne 0) { throw 'The existing .venv requires Python 3.10, 3.11, or 3.12.' }
& $python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'Unable to update pip.' }

$torchIndex = if ($env:VOXCPM_TORCH_INDEX_URL) {
    $env:VOXCPM_TORCH_INDEX_URL
} else {
    'https://download.pytorch.org/whl/cu128'
}

Write-Host "Installing PyTorch from $torchIndex"
& $python -m pip install torch torchaudio --index-url $torchIndex
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed.' }

Push-Location $project
try {
    & $python -m pip install -e .
} finally {
    Pop-Location
}
if ($LASTEXITCODE -ne 0) { throw 'VoxCPM installation failed.' }

Write-Host 'Setup complete. Double-click start-windows.cmd to launch VoxCPM.' -ForegroundColor Green
