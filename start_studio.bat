@echo off
title NeRF SR Studio

echo ==========================================
echo   NeRF SR Studio
echo ==========================================
echo.

REM --- Activate Anaconda environment ---
call F:\Anaconda\Scripts\activate.bat pytorch2023
if errorlevel 1 (
    echo [ERROR] Failed to activate conda env "pytorch2023"
    pause
    exit /b 1
)

echo [OK] Conda env: pytorch2023
echo.

REM --- Set torch model cache to PC user directory ---
set TORCH_HOME=C:\Users\PC\.cache\torch

REM --- Change to backend directory ---
cd /d "%~dp0web\backend"

REM --- Open browser after 3 seconds (background, non-blocking) ---
start /b cmd /c "timeout /t 3 /nobreak >nul & start http://localhost:8000"

echo [OK] Starting server, browser will open in 3s ...
echo.

REM --- Start FastAPI server (blocks this window, shows logs) ---
python main.py

echo.
echo [Server stopped] Press any key to close ...
pause >nul
