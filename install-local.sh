#!/bin/bash

# --- KONFIGURATION ---
MAMBA_DIR="_mamba"
ENV_DIR="_venv"
MAMBA_EXE="$MAMBA_DIR/micromamba"
PYTHON_VER="3.10"

# Fehler abfangen
set -e

# --- 1. MICROMAMBA DOWNLOAD ---
if [ ! -f "$MAMBA_EXE" ]; then
    echo "[INFO] Micromamba nicht gefunden. Lade herunter..."
    mkdir -p "$MAMBA_DIR"
    
    # Architektur erkennen
    OS_TYPE=$(uname)
    if [ "$OS_TYPE" == "Darwin" ]; then
        URL="https://micro.mamba.pm/api/micromamba/osx-64/latest" # Kein CUDA auf Mac
    else
        URL="https://micro.mamba.pm/api/micromamba/linux-64/latest"
    fi
    
    curl -L -o "$MAMBA_DIR/micromamba.tar.bz2" "$URL"
    
    echo "[INFO] Entpacke Micromamba..."
    tar -xf "$MAMBA_DIR/micromamba.tar.bz2" -C "$MAMBA_DIR"
    
    # Pfad anpassen (meistens liegt binary direkt in bin/)
    if [ -f "$MAMBA_DIR/bin/micromamba" ]; then
        mv "$MAMBA_DIR/bin/micromamba" "$MAMBA_DIR/"
        rm -rf "$MAMBA_DIR/bin"
    fi
    rm "$MAMBA_DIR/micromamba.tar.bz2"
fi

# --- 2. ENVIRONMENT ERSTELLEN ---
if [ ! -d "$ENV_DIR" ]; then
    echo "[INFO] Erstelle lokales Environment in $ENV_DIR..."
    ./$MAMBA_EXE create -p ./$ENV_DIR python=$PYTHON_VER -c conda-forge -y
fi

# --- 3. PYTORCH CUDA INSTALLATION ---
echo "[INFO] Prüfe/Installiere PyTorch..."
# Auf Linux installieren wir explizit CUDA Versionen via pip
# Auf Mac (Darwin) nutzen wir Standard pip (MPS Support ist dort standard)
if [ "$(uname)" == "Darwin" ]; then
    ./$MAMBA_EXE run -p ./$ENV_DIR pip install torch torchvision
else
    # Linux mit CUDA 12.1
    ./$MAMBA_EXE run -p ./$ENV_DIR pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
fi

# --- 4. REQUIREMENTS INSTALLIEREN ---
if [ -f "requirements.txt" ]; then
    echo "[INFO] Installiere restliche Requirements..."
    ./$MAMBA_EXE run -p ./$ENV_DIR pip install -r requirements.txt
fi

# --- 5. STARTEN ---
echo ""
echo "[START] Starte SciMind 2.0 (Lokal)..."
echo ""

export WEB_OR_LOCAL=local
./$MAMBA_EXE run -p ./$ENV_DIR uvicorn app:app --host 0.0.0.0 --port 7860 --ws websockets --reload