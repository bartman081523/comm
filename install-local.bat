@echo off
setlocal

:: --- CONFIGURATION ---
set "MAMBA_DIR=_mamba"
set "ENV_DIR=_venv"
set "MAMBA_EXE=%MAMBA_DIR%\micromamba.exe"
set "PYTHON_VER=3.10"

:: --- 1. MICROMAMBA DOWNLOAD (If missing) ---
if not exist "%MAMBA_EXE%" (
    echo [INFO] Micromamba not found. Downloading...
    if not exist "%MAMBA_DIR%" mkdir "%MAMBA_DIR%"
    
    :: Download Windows version
    curl -L -o "%MAMBA_DIR%\micromamba.tar.bz2" https://micro.mamba.pm/api/micromamba/win-64/latest
    
    echo [INFO] Unpacking Micromamba...
    :: Windows 10+ has tar built-in
    tar -xf "%MAMBA_DIR%\micromamba.tar.bz2" -C "%MAMBA_DIR%"
    
    :: Move exe from deep structure to main folder
    if exist "%MAMBA_DIR%\Library\bin\micromamba.exe" (
        move /Y "%MAMBA_DIR%\Library\bin\micromamba.exe" "%MAMBA_DIR%\"
        rmdir /S /Q "%MAMBA_DIR%\Library"
    )
    del "%MAMBA_DIR%\micromamba.tar.bz2"
)

:: --- 2. CREATE ENVIRONMENT ---
if not exist "%ENV_DIR%" (
    echo [INFO] Creating local environment in %ENV_DIR%...
    call "%MAMBA_EXE%" create -p "%ENV_DIR%" python=%PYTHON_VER% -c conda-forge -y
)

:: --- 3. PYTORCH CUDA INSTALLATION ---
echo [INFO] Installing PyTorch with CUDA support...
:: We explicitly fetch the CUDA 12.1 version
call "%MAMBA_EXE%" run -p "%ENV_DIR%" pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

:: --- 4. INSTALL REQUIREMENTS ---
if exist "requirements.txt" (
    echo [INFO] Installing remaining requirements...
    call "%MAMBA_EXE%" run -p "%ENV_DIR%" pip install -r requirements.txt
)

:: --- 5. LAUNCH ---
echo.
echo [START] Launching SciMind 2.0 (Local)...
echo.

:: Set env var to show session list locally
set WEB_OR_LOCAL=local

:: Start with Websocket support
call "%MAMBA_EXE%" run -p "%ENV_DIR%" uvicorn app:app --host 0.0.0.0 --port 7860 --ws websockets --reload

pause