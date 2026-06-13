@echo off
setlocal EnableExtensions
chcp 65001 >nul

set "PROJECT_DIR=F:\.VoxCPM\VoxCPM"
cd /d "%PROJECT_DIR%" || (
    echo [ERROR] Cannot enter project directory: %PROJECT_DIR%
    pause
    exit /b 1
)

npm.cmd run dev
pause
