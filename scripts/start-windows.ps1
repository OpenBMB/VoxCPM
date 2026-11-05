$ErrorActionPreference = 'Stop'

$project = Split-Path $PSScriptRoot -Parent
$python = Join-Path $project '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
    throw 'VoxCPM is not set up. Run setup-windows.cmd first.'
}

$hostAddress = if ($env:VOXCPM_HOST) { $env:VOXCPM_HOST } else { '127.0.0.1' }
$port = if ($env:VOXCPM_PORT) { $env:VOXCPM_PORT } else { '8808' }
$device = if ($env:VOXCPM_DEVICE) { $env:VOXCPM_DEVICE } else { 'auto' }
$model = if ($env:VOXCPM_MODEL) { $env:VOXCPM_MODEL } else { 'openbmb/VoxCPM2' }
$browserHost = if ($hostAddress -in @('0.0.0.0', '::')) { '127.0.0.1' } else { $hostAddress }
$url = "http://${browserHost}:$port"

$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
$env:GRADIO_ANALYTICS_ENABLED = 'False'
$env:HF_HUB_DISABLE_TELEMETRY = '1'
$env:GRADIO_TEMP_DIR = Join-Path $project 'temp'
New-Item -ItemType Directory -Force -Path $env:GRADIO_TEMP_DIR | Out-Null

Write-Host "Starting VoxCPM at $url" -ForegroundColor Cyan
Write-Host 'Models download when first used. Keep this window open; press Ctrl+C to stop.'
$browserArgs = @()
if ($env:VOXCPM_OPEN_BROWSER -ne '0') {
    $browserArgs += '--inbrowser'
}

Push-Location $project
try {
    & $python -u app.py --host $hostAddress --port $port --device $device --model-id $model --no-optimize @browserArgs
    if ($LASTEXITCODE -ne 0) { throw "VoxCPM exited with code $LASTEXITCODE." }
} finally {
    Pop-Location
}
