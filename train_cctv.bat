@echo off
title SafeCityAI - Entrainement Isole CCTV
echo ==============================================================================
echo SafeCityAI - Entrainement Isole et Securise pour Camera CCTV
echo Garanti 100%% sans ecrasement de best_image.pt
echo ==============================================================================

call ..\venv\Scripts\activate.bat
python train_cctv_isolated.py

echo.
pause
