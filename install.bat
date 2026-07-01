@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "ROOT=%~dp0"
cd /d "%ROOT%" || exit /b 1

set "VENV_DIR=.venv"
set "INSTALL_DEV=1"
set "INSTALL_TIMESTAMPS=1"
set "DOWNLOAD_MODEL=1"
set "DOWNLOAD_MS_MODELS=1"
set "DOWNLOAD_PARAKEET_MODEL=1"
set "DOWNLOAD_TIMESTAMP_MODEL=1"
set "RUN_SMOKE_CHECKS=1"
set "DRY_RUN=0"
set "TORCH_BACKEND=auto"
set "PYTORCH_INDEX_URL="
set "MODEL_ID=openbmb/VoxCPM2"
set "MODEL_DIR=models\openbmb__VoxCPM2"
set "PARAKEET_MODEL_ID=nvidia/parakeet-tdt-0.6b-v3"
set "PARAKEET_MODEL_DIR=models\nvidia__parakeet-tdt-0.6b-v3"
set "ZIPENHANCER_MODEL_DIR=models\iic__speech_zipenhancer_ans_multiloss_16k_base"
set "ASR_MODEL_DIR=models\iic__SenseVoiceSmall"
set "PYTHON_CMD="

goto parse_args

:usage
echo VoxCPM Windows installer
echo.
echo Usage:
echo   install.bat [options]
echo.
echo Options:
echo   --cuda                 Force CUDA-enabled torch/torchaudio wheels.
echo   --cpu                  Force CPU torch/torchaudio wheels.
echo   --pytorch-index-url U  Use a custom PyTorch wheel index URL.
echo   --model-id ID          Hugging Face model to download (default: openbmb/VoxCPM2).
echo   --model-dir DIR        Local model directory (default: models\openbmb__VoxCPM2).
echo   --download-parakeet    Pre-download NVIDIA Parakeet ASR (enabled by default).
echo   --skip-parakeet        Skip NVIDIA Parakeet ASR pre-download.
echo   --parakeet-model-dir D Local Parakeet ASR directory (default: models\nvidia__parakeet-tdt-0.6b-v3).
echo   --skip-models          Skip all model pre-downloads.
echo   --skip-modelscope      Skip ModelScope denoiser and ASR model pre-downloads.
echo   --skip-timestamp-model Skip stable-ts Whisper base model pre-download.
echo   --no-dev               Skip developer/test tools (installed by default).
echo   --no-timestamps        Skip stable-ts timestamp dependencies (installed by default).
echo   --no-smoke-checks      Skip import/CLI validation after install.
echo   --venv DIR             Use a different virtual environment directory.
echo   --dry-run              Print the planned actions without installing.
echo   -h, --help             Show this help.
echo.
echo Environment:
echo   PYTHON                 Optional path to python.exe, Python 3.10-3.12 required.
echo.
echo Examples:
echo   install.bat
echo   install.bat --cuda
echo   install.bat --cpu --model-dir D:\models\VoxCPM2
exit /b 0

:parse_args
if "%~1"=="" goto args_done
if /I "%~1"=="-h" goto usage
if /I "%~1"=="--help" goto usage
if /I "%~1"=="--cuda" (
    set "TORCH_BACKEND=cuda"
    shift
    goto parse_args
)
if /I "%~1"=="--cpu" (
    set "TORCH_BACKEND=cpu"
    shift
    goto parse_args
)
if /I "%~1"=="--download-model" (
    set "DOWNLOAD_MODEL=1"
    shift
    goto parse_args
)
if /I "%~1"=="--download-parakeet" (
    set "DOWNLOAD_PARAKEET_MODEL=1"
    shift
    goto parse_args
)
if /I "%~1"=="--skip-parakeet" (
    set "DOWNLOAD_PARAKEET_MODEL=0"
    shift
    goto parse_args
)
if /I "%~1"=="--skip-models" (
    set "DOWNLOAD_MODEL=0"
    set "DOWNLOAD_MS_MODELS=0"
    set "DOWNLOAD_PARAKEET_MODEL=0"
    set "DOWNLOAD_TIMESTAMP_MODEL=0"
    shift
    goto parse_args
)
if /I "%~1"=="--skip-modelscope" (
    set "DOWNLOAD_MS_MODELS=0"
    shift
    goto parse_args
)
if /I "%~1"=="--skip-timestamp-model" (
    set "DOWNLOAD_TIMESTAMP_MODEL=0"
    shift
    goto parse_args
)
if /I "%~1"=="--no-dev" (
    set "INSTALL_DEV=0"
    shift
    goto parse_args
)
if /I "%~1"=="--no-timestamps" (
    set "INSTALL_TIMESTAMPS=0"
    shift
    goto parse_args
)
if /I "%~1"=="--dry-run" (
    set "DRY_RUN=1"
    shift
    goto parse_args
)
if /I "%~1"=="--no-smoke-checks" (
    set "RUN_SMOKE_CHECKS=0"
    shift
    goto parse_args
)
if /I "%~1"=="--venv" goto parse_venv
if /I "%~1"=="--pytorch-index-url" goto parse_pytorch_index
if /I "%~1"=="--model-id" goto parse_model_id
if /I "%~1"=="--model-dir" goto parse_model_dir
if /I "%~1"=="--parakeet-model-dir" goto parse_parakeet_model_dir

echo Unknown option: %~1
echo Run install.bat --help for usage.
exit /b 2

:parse_venv
shift
if "%~1"=="" goto arg_error
set "VENV_DIR=%~1"
shift
goto parse_args

:parse_pytorch_index
shift
if "%~1"=="" goto arg_error
set "PYTORCH_INDEX_URL=%~1"
set "TORCH_BACKEND=custom"
shift
goto parse_args

:parse_model_id
shift
if "%~1"=="" goto arg_error
set "MODEL_ID=%~1"
shift
goto parse_args

:parse_model_dir
shift
if "%~1"=="" goto arg_error
set "MODEL_DIR=%~1"
shift
goto parse_args

:parse_parakeet_model_dir
shift
if "%~1"=="" goto arg_error
set "PARAKEET_MODEL_DIR=%~1"
shift
goto parse_args

:arg_error
echo Missing value for the previous option.
echo Run install.bat --help for usage.
exit /b 2

:args_done
if "%INSTALL_TIMESTAMPS%"=="0" set "DOWNLOAD_TIMESTAMP_MODEL=0"
if /I "%TORCH_BACKEND%"=="auto" (
    call :detect_torch_backend
)
if /I "%TORCH_BACKEND%"=="cuda" if not defined PYTORCH_INDEX_URL set "PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu121"
if /I "%TORCH_BACKEND%"=="cpu" if not defined PYTORCH_INDEX_URL set "PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu"
if /I "%DOWNLOAD_PARAKEET_MODEL%"=="auto" set "DOWNLOAD_PARAKEET_MODEL=1"

set "PROJECT_SPEC=."
if "%INSTALL_TIMESTAMPS%"=="1" if "%INSTALL_DEV%"=="1" set "PROJECT_SPEC=.[timestamps,dev]"
if "%INSTALL_TIMESTAMPS%"=="1" if "%INSTALL_DEV%"=="0" set "PROJECT_SPEC=.[timestamps]"
if "%INSTALL_TIMESTAMPS%"=="0" if "%INSTALL_DEV%"=="1" set "PROJECT_SPEC=.[dev]"

echo.
echo VoxCPM setup
echo   Root:        %CD%
echo   Venv:        %VENV_DIR%
echo   Project:     %PROJECT_SPEC%
echo   Torch:       %TORCH_BACKEND%
if defined PYTORCH_INDEX_URL echo   Torch index: %PYTORCH_INDEX_URL%
if "%DOWNLOAD_MODEL%"=="1" echo   HF model:    %MODEL_ID% -^> %MODEL_DIR%
if "%DOWNLOAD_PARAKEET_MODEL%"=="1" echo   Parakeet:    %PARAKEET_MODEL_ID% -^> %PARAKEET_MODEL_DIR%
if "%DOWNLOAD_MS_MODELS%"=="1" echo   MS models:   local denoiser + ASR models
if "%DOWNLOAD_TIMESTAMP_MODEL%"=="1" echo   TS model:    stable-ts Whisper base
if "%DRY_RUN%"=="1" echo   Mode:        dry run

call :find_python
if errorlevel 1 goto fail

echo   Python:      !PYTHON_CMD!

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo.
    echo ^> !PYTHON_CMD! -m venv "%VENV_DIR%"
    if not "%DRY_RUN%"=="1" (
        !PYTHON_CMD! -m venv "%VENV_DIR%"
        if errorlevel 1 goto fail
    )
) else (
    echo.
    echo Reusing existing virtual environment: %VENV_DIR%
)

if "%DRY_RUN%"=="1" (
    echo.
    echo ^> call "%VENV_DIR%\Scripts\activate.bat"
) else (
    call "%VENV_DIR%\Scripts\activate.bat"
    if errorlevel 1 goto fail
)

call :run python -m pip install --upgrade pip
if errorlevel 1 goto fail

call :run python -m pip install --upgrade uv
if errorlevel 1 goto fail

if exist "%VENV_DIR%\Scripts\uv.exe" (
    set "UV_CMD=%VENV_DIR%\Scripts\uv.exe"
) else (
    set "UV_CMD=uv"
)

if defined PYTORCH_INDEX_URL (
    call :run "!UV_CMD!" pip install --upgrade torch torchaudio --index-url "%PYTORCH_INDEX_URL%"
    if errorlevel 1 (
        echo.
        echo PyTorch wheel install failed; retrying with pip.
        call :run python -m pip install --upgrade torch torchaudio --index-url "%PYTORCH_INDEX_URL%"
        if errorlevel 1 goto fail
    )
)

call :run "!UV_CMD!" pip install -e "%PROJECT_SPEC%"
if errorlevel 1 (
    echo.
    echo uv install failed; retrying with pip.
    call :run python -m pip install -e "%PROJECT_SPEC%"
    if errorlevel 1 goto fail
)

if not "%DRY_RUN%"=="1" (
    if not exist "models" mkdir "models"
    if not exist "lora" mkdir "lora"
    if not exist "outputs" mkdir "outputs"
    if not exist "checkpoints" mkdir "checkpoints"
) else (
    echo.
    echo Would create runtime directories: models, lora, outputs, checkpoints
)

if not "%DOWNLOAD_MODEL%"=="1" goto skip_hf_download
echo.
echo ^> Downloading %MODEL_ID% to %MODEL_DIR%
if "%DRY_RUN%"=="1" goto skip_hf_download
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='%MODEL_ID%', local_dir=r'%MODEL_DIR%')"
if errorlevel 1 goto fail

:skip_hf_download
if not "%DOWNLOAD_PARAKEET_MODEL%"=="1" goto skip_parakeet_download
echo.
echo ^> Downloading NVIDIA Parakeet ASR to %PARAKEET_MODEL_DIR%
if "%DRY_RUN%"=="1" goto skip_parakeet_download
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='%PARAKEET_MODEL_ID%', local_dir=r'%PARAKEET_MODEL_DIR%')"
if errorlevel 1 goto fail

:skip_parakeet_download
if not "%DOWNLOAD_MS_MODELS%"=="1" goto skip_modelscope_downloads
echo.
echo ^> Downloading ModelScope denoiser to %ZIPENHANCER_MODEL_DIR%
if "%DRY_RUN%"=="1" goto skip_modelscope_asr_download
python -c "from modelscope import snapshot_download; snapshot_download('iic/speech_zipenhancer_ans_multiloss_16k_base', local_dir=r'%ZIPENHANCER_MODEL_DIR%')"
if errorlevel 1 goto fail

:skip_modelscope_asr_download
echo.
echo ^> Downloading ModelScope ASR to %ASR_MODEL_DIR%
if "%DRY_RUN%"=="1" goto skip_modelscope_downloads
python -c "from modelscope import snapshot_download; snapshot_download('iic/SenseVoiceSmall', local_dir=r'%ASR_MODEL_DIR%')"
if errorlevel 1 goto fail

:skip_modelscope_downloads
if not "%DOWNLOAD_TIMESTAMP_MODEL%"=="1" goto skip_timestamp_download
echo.
echo ^> Downloading stable-ts Whisper base model
if "%DRY_RUN%"=="1" goto skip_timestamp_download
python -c "import stable_whisper; stable_whisper.load_model('base')"
if errorlevel 1 goto fail

:skip_timestamp_download

if not "%RUN_SMOKE_CHECKS%"=="1" goto skip_smoke_checks
call :run python -m pip show voxcpm torch torchaudio gradio modelscope huggingface-hub
if errorlevel 1 goto fail
call :run python -c "import torch, torchaudio, gradio, voxcpm, soundfile, librosa, transformers, datasets, huggingface_hub, modelscope, safetensors, argbind, yaml, funasr, tensorboardX"
if errorlevel 1 goto fail
if not "%DOWNLOAD_PARAKEET_MODEL%"=="1" goto skip_parakeet_smoke
call :run python -c "from transformers import AutoModelForTDT, AutoProcessor; AutoProcessor.from_pretrained(r'%PARAKEET_MODEL_DIR%', local_files_only=True)"
if errorlevel 1 goto fail

:skip_parakeet_smoke
if not "%INSTALL_TIMESTAMPS%"=="1" goto skip_timestamp_smoke
call :run python -c "import stable_whisper"
if errorlevel 1 goto fail

:skip_timestamp_smoke
if not "%INSTALL_DEV%"=="1" goto skip_dev_smoke
call :run python -m pytest --version
if errorlevel 1 goto fail

:skip_dev_smoke
call :run voxcpm --help
if errorlevel 1 goto fail
if /I not "%TORCH_BACKEND%"=="cuda" goto skip_smoke_checks
call :verify_cuda
if errorlevel 1 goto fail

:skip_smoke_checks

echo.
echo Install complete.
echo.
set "RUNTIME_DEVICE_ARG="
if /I "%TORCH_BACKEND%"=="cuda" set "RUNTIME_DEVICE_ARG= --device cuda"
if /I "%TORCH_BACKEND%"=="cpu" set "RUNTIME_DEVICE_ARG= --device cpu"
echo Start commands:
echo   %VENV_DIR%\Scripts\activate.bat
echo   python app.py --model-id "%MODEL_DIR%" --port 8808%RUNTIME_DEVICE_ARG% --asr-backend auto
echo   voxcpm --help
echo   voxcpm design --model-path "%MODEL_DIR%"%RUNTIME_DEVICE_ARG% --text "Hello from VoxCPM2." --output outputs\demo.wav
echo   python lora_ft_webui.py
echo.
echo Notes:
echo   Web demo, CLI, and LoRA fine-tuning UI are installed.
if "%INSTALL_TIMESTAMPS%"=="1" echo   Timestamp dependencies are installed.
if "%DOWNLOAD_MODEL%"=="1" echo   Default Hugging Face model is installed at %MODEL_DIR%.
if "%DOWNLOAD_MODEL%"=="0" echo   Hugging Face model pre-download was skipped.
if "%DOWNLOAD_PARAKEET_MODEL%"=="1" echo   NVIDIA Parakeet ASR is installed at %PARAKEET_MODEL_DIR%.
if "%DOWNLOAD_PARAKEET_MODEL%"=="0" echo   NVIDIA Parakeet ASR pre-download was skipped.
if "%DOWNLOAD_MS_MODELS%"=="1" echo   Local ModelScope denoiser and ASR models are installed under models.
if "%DOWNLOAD_TIMESTAMP_MODEL%"=="1" echo   stable-ts Whisper base model was cached.
echo   CUDA is selected automatically when an NVIDIA GPU is detected; use --cpu to force CPU wheels.
exit /b 0

:detect_torch_backend
where nvidia-smi >nul 2>nul
if not errorlevel 1 (
    set "TORCH_BACKEND=cuda"
    exit /b 0
)

powershell -NoProfile -ExecutionPolicy Bypass -Command "$gpus = Get-CimInstance Win32_VideoController; foreach ($gpu in $gpus) { if ($gpu.Name -match 'NVIDIA') { exit 0 } }; exit 1" >nul 2>nul
if not errorlevel 1 (
    set "TORCH_BACKEND=cuda"
    exit /b 0
)

wmic path win32_VideoController get name 2>nul | findstr /I "NVIDIA" >nul 2>nul
if not errorlevel 1 (
    set "TORCH_BACKEND=cuda"
    exit /b 0
)

set "TORCH_BACKEND=cpu"
exit /b 0

:find_python
if defined PYTHON (
    "%PYTHON%" -c "import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 13) else 1)" >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_CMD="%PYTHON%""
        exit /b 0
    )
)

py -3.12 -c "import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 13) else 1)" >nul 2>nul
if not errorlevel 1 (
    set "PYTHON_CMD=py -3.12"
    exit /b 0
)

py -3.11 -c "import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 13) else 1)" >nul 2>nul
if not errorlevel 1 (
    set "PYTHON_CMD=py -3.11"
    exit /b 0
)

py -3.10 -c "import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 13) else 1)" >nul 2>nul
if not errorlevel 1 (
    set "PYTHON_CMD=py -3.10"
    exit /b 0
)

python -c "import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 13) else 1)" >nul 2>nul
if not errorlevel 1 (
    set "PYTHON_CMD=python"
    exit /b 0
)

echo.
echo Python 3.10, 3.11, or 3.12 was not found.
echo Install Python from https://www.python.org/downloads/windows/ and rerun install.bat.
exit /b 1

:verify_cuda
echo.
echo ^> python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"
if "%DRY_RUN%"=="1" exit /b 0
python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"
if errorlevel 1 (
    echo CUDA torch wheels were installed, but torch CUDA availability check returned false.
    echo Check the NVIDIA driver, or rerun install.bat --cpu for CPU-only setup.
)
exit /b %ERRORLEVEL%

:run
echo.
echo ^> %*
if "%DRY_RUN%"=="1" exit /b 0
%*
exit /b %ERRORLEVEL%

:fail
echo.
echo Installation failed. See the error above.
exit /b 1
