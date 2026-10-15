@echo off
title Safety AI - Launching Dashboard and API
echo ===================================================
echo               Safety AI Deployment
echo ===================================================
echo.

cd /d "%~dp0"

:: 1. Check if virtual environment already exists locally or in parent
set "PY_CMD="
if exist "venv\Scripts\python.exe" (
    set "PY_CMD=venv\Scripts\python.exe"
    echo [INFO] Using existing local virtual environment: .\venv
) else if exist "..\venv\Scripts\python.exe" (
    set "PY_CMD=..\venv\Scripts\python.exe"
    echo [INFO] Using existing root virtual environment: ..\venv
)

:: 2. Find system python if not in venv
if "%PY_CMD%"=="" (
    python --version >nul 2>&1
    if %errorlevel% equ 0 (
        set "PY_CMD=python"
    ) else (
        py -3 --version >nul 2>&1
        if %errorlevel% equ 0 (
            set "PY_CMD=py -3"
        ) else if exist "%LocalAppData%\Programs\Python\Python310\python.exe" (
            set "PY_CMD=%LocalAppData%\Programs\Python\Python310\python.exe"
        ) else if exist "C:\Python310\python.exe" (
            set "PY_CMD=C:\Python310\python.exe"
        )
    )
)

if "%PY_CMD%"=="" (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python (>= 3.8) and try again.
    pause
    exit /b 1
)

:: 3. Activate virtual environment if available
if exist "venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment...
    call "venv\Scripts\activate.bat"
    set "PY_CMD=python"
) else if exist "..\venv\Scripts\activate.bat" (
    echo [INFO] Activating root virtual environment...
    call "..\venv\Scripts\activate.bat"
    set "PY_CMD=python"
) else (
    echo [INFO] Creating local virtual environment in .\venv...
    %PY_CMD% -m venv venv
    if exist "venv\Scripts\activate.bat" (
        call "venv\Scripts\activate.bat"
        set "PY_CMD=python"
        echo [INFO] Installing required dependencies...
        python -m pip install --upgrade pip
        python -m pip install -r requirements.txt
    )
)

:: Generate demo video if not present
if not exist "output_violation_demo.mp4" (
    echo [INFO] Generating demo video...
    %PY_CMD% demo_video.py
)

:: Run Server
echo.
echo ===================================================
echo  Safety AI Server starting on http://127.0.0.1:5000
echo ===================================================
echo.
%PY_CMD% server.py

pause
