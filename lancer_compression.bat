@echo off
title SafeCityAI - Packaging Dataset Deliverable
echo ===================================================
echo            SafeCityAI Dataset Packaging
echo ===================================================
echo.

cd /d "%~dp0"

if exist "..\venv\Scripts\activate.bat" (
    call "..\venv\Scripts\activate.bat"
) else if exist "venv\Scripts\activate.bat" (
    call "venv\Scripts\activate.bat"
)

python zip_dataset.py
echo.
echo ===================================================
echo Packaging complete. Press any key to close.
echo ===================================================
pause
