#!/usr/bin/env bash
#
# One-click setup and run script for the custom EmotionNet backend (macOS/Linux).
#
# This script does everything needed to get the backend running:
#   1. Creates a Python virtual environment (if it doesn't exist)
#   2. Activates the venv
#   3. Installs all dependencies from requirements.txt
#   4. Copies .env.example → .env (if .env doesn't exist)
#   5. Trains the model (if emotion_model.pth doesn't exist)
#   6. Starts the server on port 8000
#
# Usage:
#   cd backend
#   chmod +x setup.sh
#   ./setup.sh

set -e

# ── Ensure we're in the right directory ──────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "============================================"
echo "  EmotionNet Backend Setup (macOS/Linux)"
echo "============================================"
echo ""

# ── Step 1: Check Python ────────────────────────────────────────────────────
echo "[1/6] Checking Python..."

PYTHON_CMD=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        ver=$("$cmd" --version 2>&1)
        # Extract minor version number
        minor=$(echo "$ver" | grep -oP 'Python 3\.(\d+)' | grep -oP '\d+$')
        if [ -z "$minor" ]; then
            # Fallback for macOS grep (no -P flag)
            minor=$(echo "$ver" | sed -n 's/Python 3\.\([0-9]*\).*/\1/p')
        fi
        if [ -n "$minor" ] && [ "$minor" -ge 10 ]; then
            PYTHON_CMD="$cmd"
            echo "      Found: $ver"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "      ERROR: Python 3.10+ is required but not found."
    echo "      Install from: https://www.python.org/downloads/"
    exit 1
fi

# ── Step 2: Create virtual environment ───────────────────────────────────────
echo "[2/6] Setting up virtual environment..."

if [ ! -d "venv" ]; then
    echo "      Creating venv..."
    "$PYTHON_CMD" -m venv venv
    echo "      Created venv/ directory."
else
    echo "      venv/ already exists — reusing."
fi

# Activate the venv
# shellcheck disable=SC1091
source venv/bin/activate
echo "      Activated virtual environment."

# ── Step 3: Install dependencies ─────────────────────────────────────────────
echo "[3/6] Installing dependencies..."
echo "      This may take 5-10 minutes on first run (PyTorch is ~2 GB)."

pip install --upgrade pip --quiet 2>/dev/null
pip install -r requirements.txt --quiet

echo "      All dependencies installed."

# ── Step 4: Create .env (if needed) ──────────────────────────────────────────
echo "[4/6] Checking .env configuration..."

if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "      Created .env from .env.example."
        echo "      (Optional) Edit .env to add your GEMINI_API_KEY for AI reports."
    else
        echo "      No .env.example found — skipping."
    fi
else
    echo "      .env already exists — keeping your configuration."
fi

# ── Step 5: Train the model ─────────────────────────────────────────────────
echo "[5/6] Checking model..."

if [ ! -f "emotion_model.pth" ]; then
    # Check that training data exists
    if [ ! -d "processed_data" ]; then
        echo "      ERROR: processed_data/ directory not found."
        echo "      Place the training images in backend/processed_data/ with"
        echo "      subdirectories: angry/ disgust/ fear/ happy/ neutral/ sad/ surprise/"
        exit 1
    fi

    echo "      No trained model found. Training is required."
    echo ""
    echo "      Training will take 10-60 minutes depending on your hardware."
    echo "      (If you want to edit .env first, press Ctrl+C now, edit it, then re-run this script.)"
    echo ""

    read -rp "      Start training now? (Y/n) " response
    if [ "$response" = "n" ] || [ "$response" = "N" ]; then
        echo "      Skipping training. Run 'python train.py' when you're ready."
        echo "      Then run 'uvicorn server:app --host 0.0.0.0 --port 8000 --reload' to start the server."
        exit 0
    fi

    echo ""
    echo "      Training started..."
    echo ""
    "$PYTHON_CMD" train.py

    if [ ! -f "emotion_model.pth" ]; then
        echo "      ERROR: Training completed but emotion_model.pth was not created."
        exit 1
    fi

    echo ""
    echo "      Model trained and saved to emotion_model.pth"
else
    echo "      emotion_model.pth found — skipping training."
    echo "      (To retrain, delete emotion_model.pth and run this script again.)"
fi

# ── Step 6: Start the server ────────────────────────────────────────────────
echo "[6/6] Starting the server..."
echo ""
echo "============================================"
echo "  Server starting on http://localhost:8000"
echo "  Health check: http://localhost:8000/health"
echo "  WebSocket:    ws://localhost:8000/ws/{id}"
echo "  Press Ctrl+C to stop."
echo "============================================"
echo ""

uvicorn server:app --host 0.0.0.0 --port 8000 --reload
