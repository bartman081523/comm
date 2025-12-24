#!/bin/bash

# --- CONFIGURATION ---
MAMBA_DIR="_mamba"
ENV_DIR="_venv"
MAMBA_EXE="$MAMBA_DIR/micromamba"
PYTHON_VER="3.10"

# Stop on error
set -e

# --- 1. MICROMAMBA DOWNLOAD ---
if [ ! -f "$MAMBA_EXE" ]; then
    echo "[INFO] Micromamba not found. Downloading..."
    mkdir -p "$MAMBA_DIR"
    
    # Detect Architecture
    OS_TYPE=$(uname)
    if [ "$OS_TYPE" == "Darwin" ]; then
        URL="https://micro.mamba.pm/api/micromamba/osx-64/latest" # No CUDA on Mac usually
    else
        URL="https://micro.mamba.pm/api/micromamba/linux-64/latest"
    fi
    
    curl -L -o "$MAMBA_DIR/micromamba.tar.bz2" "$URL"
    
    echo "[INFO] Unpacking Micromamba..."
    tar -xf "$MAMBA_DIR/micromamba.tar.bz2" -C "$MAMBA_DIR"
    
    # Adjust path (binary usually in bin/)
    if [ -f "$MAMBA_DIR/bin/micromamba" ]; then
        mv "$MAMBA_DIR/bin/micromamba" "$MAMBA_DIR/"
        rm -rf "$MAMBA_DIR/bin"
    fi
    rm "$MAMBA_DIR/micromamba.tar.bz2"
fi

# --- 2. CREATE ENVIRONMENT ---
if [ ! -d "$ENV_DIR" ]; then
    echo "[INFO] Creating local environment in $ENV_DIR..."
    ./$MAMBA_EXE create -p ./$ENV_DIR python=$PYTHON_VER -c conda-forge -y
fi

# --- 3. PYTORCH INSTALLATION ---
echo "[INFO] Checking/Installing PyTorch..."
if [ "$(uname)" == "Darwin" ]; then
    # Mac (MPS support is standard in pip)
    ./$MAMBA_EXE run -p ./$ENV_DIR pip install torch torchvision
else
    # Linux with CUDA 12.1
    ./$MAMBA_EXE run -p ./$ENV_DIR pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
fi

# --- 4. INSTALL REQUIREMENTS ---
if [ -f "requirements.txt" ]; then
    echo "[INFO] Installing remaining requirements..."
    ./$MAMBA_EXE run -p ./$ENV_DIR pip install -r requirements.txt
fi

# --- 5. LAUNCH ---
echo ""
echo "[START] Launching SciMind 2.0 (Local)..."
echo ""

export WEB_OR_LOCAL=local
./$MAMBA_EXE run -p ./$ENV_DIR uvicorn app:app --host 0.0.0.0 --port 7860 --ws websockets --reload