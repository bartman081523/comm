@echo off
setlocal

:: --- KONFIGURATION ---
set "MAMBA_DIR=_mamba"
set "ENV_DIR=_venv"
set "MAMBA_EXE=%MAMBA_DIR%\micromamba.exe"
set "PYTHON_VER=3.10"

:: --- 1. MICROMAMBA DOWNLOAD (Falls nicht vorhanden) ---
if not exist "%MAMBA_EXE%" (
    echo [INFO] Micromamba nicht gefunden. Lade herunter...
    if not exist "%MAMBA_DIR%" mkdir "%MAMBA_DIR%"
    
    :: Download der Windows-Version
    curl -L -o "%MAMBA_DIR%\micromamba.tar.bz2" https://micro.mamba.pm/api/micromamba/win-64/latest
    
    echo [INFO] Entpacke Micromamba...
    :: Windows hat seit Win10 tar an Bord
    tar -xf "%MAMBA_DIR%\micromamba.tar.bz2" -C "%MAMBA_DIR%"
    
    :: Verschiebe die exe aus der tiefen Struktur nach oben
    if exist "%MAMBA_DIR%\Library\bin\micromamba.exe" (
        move /Y "%MAMBA_DIR%\Library\bin\micromamba.exe" "%MAMBA_DIR%\"
        rmdir /S /Q "%MAMBA_DIR%\Library"
    )
    del "%MAMBA_DIR%\micromamba.tar.bz2"
)

:: --- 2. ENVIRONMENT ERSTELLEN ---
if not exist "%ENV_DIR%" (
    echo [INFO] Erstelle lokales Environment in %ENV_DIR%...
    call "%MAMBA_EXE%" create -p "%ENV_DIR%" python=%PYTHON_VER% -c conda-forge -y
)

:: --- 3. PYTORCH CUDA INSTALLATION ---
echo [INFO] Installiere PyTorch mit CUDA Support...
:: Wir nutzen pip im Environment. Wir holen die CUDA 12.1 Version explizit.
call "%MAMBA_EXE%" run -p "%ENV_DIR%" pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

:: --- 4. REQUIREMENTS INSTALLIEREN ---
if exist "requirements.txt" (
    echo [INFO] Installiere restliche Requirements...
    call "%MAMBA_EXE%" run -p "%ENV_DIR%" pip install -r requirements.txt
)

:: --- 5. STARTEN ---
echo.
echo [START] Starte SciMind 2.0 (Lokal)...
echo.

:: Umgebungsvariable setzen, damit Session-Liste angezeigt wird
set WEB_OR_LOCAL=local

:: Start mit Websockets Unterstützung
call "%MAMBA_EXE%" run -p "%ENV_DIR%" uvicorn app:app --host 0.0.0.0 --port 7860 --ws websockets --reload

pause