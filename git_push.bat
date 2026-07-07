@echo off
title SafeCityAI - Git Push Automation
echo ===================================================
echo               SafeCityAI Git Commit & Push
echo ===================================================
echo.

cd /d "%~dp0"

:: Check if git is installed
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Git is not installed or not in PATH.
    pause
    exit /b 1
)

:: Show current changes status
echo [INFO] Git status:
git status
echo.

:: Add files
echo [INFO] Staging all files...
git add .

:: Commit files
set /p commit_msg="Enter commit message [Default: 'Complete SafeCityAI Deliverables']: "
if "%commit_msg%"=="" set commit_msg="Complete SafeCityAI Deliverables"

echo [INFO] Committing changes...
git commit -m "%commit_msg%"

:: Push to remote repository
echo [INFO] Pushing to remote repository...
git push

echo.
echo ===================================================
echo Git process finished. Press any key to close.
echo ===================================================
pause
