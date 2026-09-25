@echo off
title SafeCityAI - Launching Dashboard and API
echo ===================================================
echo               SafeCityAI Deployment
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

:: Generate the 30-second demo video if it doesn't exist
if not exist "output_violation_demo.mp4" (
    echo [INFO] Generating demo video (30 seconds)...
    python demo_video.py
)

:: Generate gallery test images if script exists
if exist "static\samples\generate_samples.py" (
    echo [INFO] Generating quick test gallery images...
    python static/samples/generate_samples.py
)

echo.
echo [SUCCESS] SafeCityAI is ready!
echo [INFO] Starting Flask Server...
echo [INFO] Access the dashboard in your browser: http://localhost:5000
echo.
echo Press Ctrl+C in this terminal to stop the server.
echo ===================================================
python server.py
pause