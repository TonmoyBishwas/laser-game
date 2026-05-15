@echo off
REM ============================================================================
REM  Laser-Guard uni setup script
REM  Run this ONCE on the friend's laptop after cloning the GitHub repo.
REM  Creates a Python venv and installs CPU-only dependencies (~200 MB download).
REM ============================================================================

setlocal

echo.
echo [setup] checking python...
where python >nul 2>nul
if errorlevel 1 (
    echo [setup] ERROR: python is not on PATH.
    echo         Install Python 3.10+ from https://www.python.org/downloads/
    echo         Tick "Add python.exe to PATH" during install, then re-run this script.
    pause
    exit /b 1
)

python --version

echo.
echo [setup] creating venv at .\venv ...
python -m venv venv
if errorlevel 1 (
    echo [setup] ERROR: venv creation failed.
    pause
    exit /b 1
)

echo.
echo [setup] upgrading pip...
venv\Scripts\python.exe -m pip install --upgrade pip

echo.
echo [setup] installing CPU-only torch + the rest (this is the big download)...
venv\Scripts\python.exe -m pip install ^
    --index-url https://download.pytorch.org/whl/cpu ^
    --extra-index-url https://pypi.org/simple ^
    -r requirements_uni.txt
if errorlevel 1 (
    echo [setup] ERROR: pip install failed.
    pause
    exit /b 1
)

echo.
echo [setup] verifying torch can import...
venv\Scripts\python.exe -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())"

echo.
echo [setup] DONE.
echo.
echo Next steps:
echo   1. Plug in your phone (USB), open DroidCam on both phone + Windows app, click "USB".
echo   2. Run a camera scan:        venv\Scripts\python.exe find_cam.py
echo   3. Run the game (CPU mode):  venv\Scripts\python.exe -m game.main --profile uni --recalibrate --debug
echo.
endlocal
pause
