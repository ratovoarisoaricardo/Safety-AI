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
    echo Please install Python and try again.
    pause
    exit /b 1
)

:: Try activating the virtual environment in the parent folder if it exists
if exist "..\venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment found in parent directory...
    call "..\venv\Scripts\activate.bat"
) else if exist "venv\Scripts\activate.bat" (
    echo [INFO] Activating local virtual environment...
    call "venv\Scripts\activate.bat"
)

:: Install dependencies
echo [INFO] Installing required dependencies from requirements.txt...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

:: Generate the 30-second demo video if it doesn't exist
echo [INFO] Generating demo video (30 seconds)...
python demo_video.py

:: Generate the 30 gallery test images
echo [INFO] Generating quick test gallery images...
python static/samples/generate_samples.py

echo.
echo [SUCCESS] Dependencies verified.
echo [INFO] Starting Flask Server...
echo [INFO] Access the dashboard in your browser: http://localhost:5000
echo.
echo Press Ctrl+C in this terminal to stop the server.
echo ===================================================
python server.py
pause
