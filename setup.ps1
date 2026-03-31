<#
.SYNOPSIS
    One-click setup and run script for the custom EmotionNet backend (Windows).

.DESCRIPTION
    This script does everything needed to get the backend running:
      1. Creates a Python virtual environment (if it doesn't exist)
      2. Activates the venv
      3. Installs all dependencies from requirements.txt
      4. Copies .env.example → .env (if .env doesn't exist)
      5. Trains the model (if emotion_model.pth doesn't exist)
      6. Starts the server on port 8000

.NOTES
    Run from the backend/ directory:
        .\setup.ps1

    If you get an execution policy error, run this first (one-time):
        Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
#>

$ErrorActionPreference = "Stop"

# ── Ensure we're in the right directory ──────────────────────────────────────
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  EmotionNet Backend Setup (Windows)" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# ── Step 1: Check Python ─────────────────────────────────────────────────────
Write-Host "[1/6] Checking Python..." -ForegroundColor Yellow

$PythonCmd = $null
foreach ($cmd in @("python", "python3", "py")) {
    try {
        $ver = & $cmd --version 2>&1
        if ($ver -match "Python 3\.(\d+)") {
            $minor = [int]$Matches[1]
            if ($minor -ge 10) {
                $PythonCmd = $cmd
                Write-Host "      Found: $ver" -ForegroundColor Green
                break
            }
        }
    } catch { }
}

if (-not $PythonCmd) {
    Write-Host "      ERROR: Python 3.10+ is required but not found." -ForegroundColor Red
    Write-Host "      Download from: https://www.python.org/downloads/" -ForegroundColor Red
    Write-Host "      Make sure to check 'Add Python to PATH' during installation." -ForegroundColor Red
    exit 1
}

# ── Step 2: Create virtual environment ────────────────────────────────────────
Write-Host "[2/6] Setting up virtual environment..." -ForegroundColor Yellow

if (-not (Test-Path "venv")) {
    Write-Host "      Creating venv..." -ForegroundColor Gray
    & $PythonCmd -m venv venv
    Write-Host "      Created venv/ directory." -ForegroundColor Green
} else {
    Write-Host "      venv/ already exists — reusing." -ForegroundColor Green
}

# Activate the venv
$ActivateScript = Join-Path $ScriptDir "venv\Scripts\Activate.ps1"
if (-not (Test-Path $ActivateScript)) {
    Write-Host "      ERROR: venv activation script not found at $ActivateScript" -ForegroundColor Red
    exit 1
}
. $ActivateScript
Write-Host "      Activated virtual environment." -ForegroundColor Green

# ── Step 3: Install dependencies ─────────────────────────────────────────────
Write-Host "[3/6] Installing dependencies..." -ForegroundColor Yellow
Write-Host "      This may take 5-10 minutes on first run (PyTorch is ~2 GB)." -ForegroundColor Gray

pip install --upgrade pip --quiet 2>$null
pip install -r requirements.txt --quiet

if ($LASTEXITCODE -ne 0) {
    Write-Host "      ERROR: pip install failed. Check the output above." -ForegroundColor Red
    exit 1
}
Write-Host "      All dependencies installed." -ForegroundColor Green

# ── Step 4: Create .env (if needed) ──────────────────────────────────────────
Write-Host "[4/6] Checking .env configuration..." -ForegroundColor Yellow

if (-not (Test-Path ".env")) {
    if (Test-Path ".env.example") {
        Copy-Item ".env.example" ".env"
        Write-Host "      Created .env from .env.example." -ForegroundColor Green
        Write-Host "      (Optional) Edit .env to add your GEMINI_API_KEY for AI reports." -ForegroundColor Gray
    } else {
        Write-Host "      No .env.example found — skipping." -ForegroundColor Gray
    }
} else {
    Write-Host "      .env already exists — keeping your configuration." -ForegroundColor Green
}

# ── Step 5: Train the model ──────────────────────────────────────────────────
Write-Host "[5/6] Checking model..." -ForegroundColor Yellow

if (-not (Test-Path "emotion_model.pth")) {
    # Check that training data exists
    if (-not (Test-Path "processed_data")) {
        Write-Host "      ERROR: processed_data/ directory not found." -ForegroundColor Red
        Write-Host "      Place the training images in backend/processed_data/ with" -ForegroundColor Red
        Write-Host "      subdirectories: angry/ disgust/ fear/ happy/ neutral/ sad/ surprise/" -ForegroundColor Red
        exit 1
    }

    Write-Host "      No trained model found. Training is required." -ForegroundColor Gray
    Write-Host ""
    Write-Host "      Training will take 10-60 minutes depending on your hardware." -ForegroundColor Gray
    Write-Host "      (If you want to edit .env first, press Ctrl+C now, edit it, then re-run this script.)" -ForegroundColor Gray
    Write-Host ""

    $response = Read-Host "      Start training now? (Y/n)"
    if ($response -eq "n" -or $response -eq "N") {
        Write-Host "      Skipping training. Run 'python train.py' when you're ready." -ForegroundColor Yellow
        Write-Host "      Then run 'uvicorn server:app --host 0.0.0.0 --port 8000 --reload' to start the server." -ForegroundColor Yellow
        exit 0
    }

    Write-Host ""
    Write-Host "      Training started..." -ForegroundColor Cyan
    Write-Host ""
    & $PythonCmd train.py

    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "      ERROR: Training failed. Check the output above." -ForegroundColor Red
        exit 1
    }

    if (-not (Test-Path "emotion_model.pth")) {
        Write-Host "      ERROR: Training completed but emotion_model.pth was not created." -ForegroundColor Red
        exit 1
    }

    Write-Host ""
    Write-Host "      Model trained and saved to emotion_model.pth" -ForegroundColor Green
} else {
    Write-Host "      emotion_model.pth found — skipping training." -ForegroundColor Green
    Write-Host "      (To retrain, delete emotion_model.pth and run this script again.)" -ForegroundColor Gray
}

# ── Step 6: Start the server ─────────────────────────────────────────────────
Write-Host "[6/6] Starting the server..." -ForegroundColor Yellow
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Server starting on http://localhost:8000" -ForegroundColor Cyan
Write-Host "  Health check: http://localhost:8000/health" -ForegroundColor Cyan
Write-Host "  WebSocket:    ws://localhost:8000/ws/{id}" -ForegroundColor Cyan
Write-Host "  Press Ctrl+C to stop." -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

uvicorn server:app --host 0.0.0.0 --port 8000 --reload
