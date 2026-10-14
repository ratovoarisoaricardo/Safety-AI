@echo off
title Safety AI - Launching Dashboard and API
echo ===================================================
echo               Safety AI Deployment
echo ===================================================
echo.

cd /d "%~dp0"

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python (>= 3.8) and try again.
    pause
    exit /b 1
)

:: Create or activate local virtual environment
if not exist "venv\Scripts\activate.bat" (
    echo [INFO] Creating local virtual environment in .\venv...
    python -m venv venv
)

echo [INFO] Activating virtual environment...
call "venv\Scripts\activate.bat"

:: Install dependencies
echo [INFO] Installing required dependencies from requirements.txt...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

:: Generate the demo video if it doesn't exist
if not exist "output_violation_demo.mp4" (
    echo [INFO] Generating demo video...
    python demo_video.py
)

:: Run Server
echo.
echo ===================================================
echo  Safety AI Server starting on http://127.0.0.1:5000
echo ===================================================
echo.
python server.py

pause
