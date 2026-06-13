@echo off
setlocal EnableExtensions
chcp 65001 >nul

set "PROJECT_DIR=F:\.VoxCPM\VoxCPM"
set "MAIN_PORT=8808"
set "LORA_PORT=7860"

cd /d "%PROJECT_DIR%" || (
    echo.
    echo [ERROR] Cannot enter project directory:
    echo %PROJECT_DIR%
    echo.
    pause
    exit /b 1
)

call :activate_venv

:menu
cls
echo ================================================
echo              VoxCPM Windows Launcher
echo ================================================
echo Project: %PROJECT_DIR%
echo.
echo Recommended: [1] Main WebUI - GPU CUDA
echo.
echo   1. Start Main WebUI - GPU CUDA
echo   2. Start Main WebUI - Auto Device
echo   3. Start Main WebUI - CPU
echo   4. Start LoRA Training/Inference WebUI
echo   5. Check CUDA/GPU and Python
echo   6. Show install commands
echo   7. Show common CLI examples
echo   8. Exit
echo   9. Open Electron App Shell
echo.
set /p "CHOICE=Choose a mode [1-9]: "
set "CHOICE=%CHOICE: =%"

if not defined CHOICE goto invalid_choice
if "%CHOICE%"=="1" goto main_cuda
if "%CHOICE%"=="2" goto main_auto
if "%CHOICE%"=="3" goto main_cpu
if "%CHOICE%"=="4" goto lora_webui
if "%CHOICE%"=="5" goto check_env
if "%CHOICE%"=="6" goto install_help
if "%CHOICE%"=="7" goto cli_examples
if "%CHOICE%"=="8" goto end
if "%CHOICE%"=="9" goto dev_shell

:invalid_choice
echo.
echo Invalid choice: %CHOICE%
pause
goto menu

:activate_venv
if exist "%PROJECT_DIR%\.venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment: .venv
    call "%PROJECT_DIR%\.venv\Scripts\activate.bat"
) else (
    echo [INFO] No .venv found. Using current Python environment.
    echo [INFO] Use menu option 6 to see recommended install commands.
    echo.
)
exit /b 0

:main_cuda
call :print_header "Main WebUI - GPU CUDA"
echo URL: http://localhost:%MAIN_PORT%
echo Command: python run_with_local_ffmpeg.py app.py --port %MAIN_PORT% --device cuda
echo.
python run_with_local_ffmpeg.py app.py --port %MAIN_PORT% --device cuda
echo.
pause
goto menu

:main_auto
call :print_header "Main WebUI - Auto Device"
echo URL: http://localhost:%MAIN_PORT%
echo Command: python run_with_local_ffmpeg.py app.py --port %MAIN_PORT% --device auto
echo.
python run_with_local_ffmpeg.py app.py --port %MAIN_PORT% --device auto
echo.
pause
goto menu

:main_cpu
call :print_header "Main WebUI - CPU"
echo URL: http://localhost:%MAIN_PORT%
echo Command: python run_with_local_ffmpeg.py app.py --port %MAIN_PORT% --device cpu
echo.
python run_with_local_ffmpeg.py app.py --port %MAIN_PORT% --device cpu
echo.
pause
goto menu

:lora_webui
call :print_header "LoRA Training/Inference WebUI"
echo URL: http://localhost:%LORA_PORT%
echo Command: python lora_ft_webui.py
echo.
python lora_ft_webui.py
echo.
pause
goto menu

:check_env
call :print_header "CUDA/GPU and Python Check"
echo [Python]
python --version
echo.
echo [NVIDIA GPU]
nvidia-smi
echo.
pause
goto menu

:install_help
call :print_header "Recommended Install Commands"
echo This launcher does not install dependencies automatically.
echo Run these commands manually if you want a local virtual environment:
echo.
echo   cd /d %PROJECT_DIR%
echo   python -m venv .venv
echo   .venv\Scripts\activate.bat
echo   python -m pip install --upgrade pip
echo   pip install -e .
echo.
echo After installation, run this launcher again and choose mode 1.
echo.
pause
goto menu

:cli_examples
call :print_header "Common CLI Examples"
echo Voice design:
echo   voxcpm design --text "VoxCPM2 brings studio-quality multilingual speech synthesis." --output out.wav
echo.
echo Voice design with control:
echo   voxcpm design --text "VoxCPM2 brings studio-quality multilingual speech synthesis." --control "Young female voice, warm and gentle, slightly smiling" --output out.wav
echo.
echo Voice cloning with reference audio:
echo   voxcpm clone --text "This is a voice cloning demo." --reference-audio path\to\voice.wav --output out.wav
echo.
echo Batch processing:
echo   voxcpm batch --input examples\input.txt --output-dir outs
echo.
pause
goto menu

:dev_shell
call :print_header "VoxCPM Electron App Shell"
echo Command: wscript.exe start_electron_shell.vbs
echo.
wscript.exe start_electron_shell.vbs
goto menu

:print_header
cls
echo ================================================
echo %~1
echo ================================================
echo.
exit /b 0

:end
echo.
echo Bye.
exit /b 0
