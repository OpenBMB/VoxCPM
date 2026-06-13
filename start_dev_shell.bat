@echo off
setlocal EnableExtensions
chcp 65001 >nul

set "PROJECT_DIR=F:\.VoxCPM\VoxCPM"
cd /d "%PROJECT_DIR%" || (
    echo [ERROR] Cannot enter project directory: %PROJECT_DIR%
    pause
    exit /b 1
)

if exist ".venv\Scripts\python.exe" (
    if exist ".venv\Scripts\pythonw.exe" (
        ".venv\Scripts\pythonw.exe" voxcpm_dev_shell.py
    ) else (
        ".venv\Scripts\python.exe" voxcpm_dev_shell.py
    )
) else (
    pythonw voxcpm_dev_shell.py
)

pause
