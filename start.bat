@echo off
setlocal enabledelayedexpansion

echo ========================================
echo   OpenShorts - One-Click Starter
echo ========================================
echo.

:: ── Config ──────────────────────────────
set "VENV_DIR=venv"
rem REQUIREMENTS is not used directly; deps installed together below
set "DASHBOARD_DIR=dashboard"
set "BACKEND_HOST=0.0.0.0"
set "BACKEND_PORT=8000"
set "BACKEND_WORKERS=1"

:: ── Prerequisites ───────────────────────

echo [1/6] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.10+ and try again.
    pause
    exit /b 1
)
for /f "tokens=2 delims=." %%a in ('python --version 2^>^&1') do set "PY_MINOR=%%a"
if !PY_MINOR! LSS 10 (
    echo WARNING: Python 3.10+ recommended (found 3.!PY_MINOR!)
)

echo [2/6] Checking Node.js...
node --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Node.js not found. Install Node.js 18+ and try again.
    pause
    exit /b 1
)

echo [3/6] Checking FFmpeg...
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo   FFmpeg not found in PATH. Checking local install...
    if exist "ffmpeg\bin\ffmpeg.exe" (
        echo   Found ffmpeg\bin\ffmpeg.exe. Adding to PATH...
        set "PATH=%CD%\ffmpeg\bin;%PATH%"
    ) else (
        echo   Attempting auto-download via PowerShell...
        powershell -ExecutionPolicy Bypass -File "install-ffmpeg.ps1" >nul 2>&1
        if exist "ffmpeg\bin\ffmpeg.exe" (
            echo   FFmpeg downloaded and installed to ffmpeg\bin\
            set "PATH=%CD%\ffmpeg\bin;%PATH%"
        ) else (
            echo   WARNING: Could not install FFmpeg automatically.
            echo   Clip cutting and subtitle burning will fail.
            echo   Install manually from https://ffmpeg.org/download.html
            echo.
        )
    )
) else (
    echo   FFmpeg found.
)

:: ── Python Virtual Environment ──────────

echo [4/6] Setting up Python venv...
if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo   Creating virtual environment...
    python -m venv %VENV_DIR%
    if errorlevel 1 (
        echo ERROR: Failed to create venv.
        pause
        exit /b 1
    )
    echo   Virtual environment created.
) else (
    echo   Virtual environment exists.
)

call "%VENV_DIR%\Scripts\activate.bat"

:: ── Python Dependencies ─────────────────

echo [5/6] Installing Python dependencies...
pip install -r requirements.txt -r requirements_engine.txt --quiet
if errorlevel 1 (
    echo   First pass had issues. Retrying for details...
    pip install -r requirements.txt -r requirements_engine.txt
) else (
    echo   All Python packages installed.
)

:: ── Dashboard Dependencies ──────────────

echo [6/6] Installing dashboard dependencies...
if exist "%DASHBOARD_DIR%\node_modules" (
    echo   Node modules exist. Run "npm install" in dashboard/ to update.
) else (
    echo   Installing npm packages...
    pushd "%DASHBOARD_DIR%"
    call npm install --no-audit --no-fund
    if errorlevel 1 (
        echo   WARNING: npm install had issues. Run manually: cd dashboard ^&^& npm install
    ) else (
        echo   Dashboard dependencies installed.
    )
    popd
)

echo.
echo ========================================
echo   Starting servers...
echo ========================================
echo.

:: ── Start Backend ───────────────────────
echo Starting backend (uvicorn) on port %BACKEND_PORT%...
start "OpenShorts Backend" cmd /c "title OpenShorts Backend && call %VENV_DIR%\Scripts\activate.bat && uvicorn app:app --host %BACKEND_HOST% --port %BACKEND_PORT% --workers %BACKEND_WORKERS% --reload --log-level info"

:: ── Start Frontend ──────────────────────
echo Starting frontend (Vite dev server)...
start "OpenShorts Dashboard" cmd /c "title OpenShorts Dashboard && cd %DASHBOARD_DIR% && npm run dev"

echo.
echo ========================================
echo   OpenShorts is starting up!
echo ========================================
echo.
echo   Backend:  http://localhost:%BACKEND_PORT%
echo   Frontend: http://localhost:5173
echo.
echo   API Docs: http://localhost:%BACKEND_PORT%/docs
echo.
echo   Close the server windows to stop.
echo   This window will stay open for logs.
echo ========================================
echo.

:: Keep this window open
pause
