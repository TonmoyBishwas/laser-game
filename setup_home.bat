@echo off
REM Home/dev setup — CUDA torch. Run once after cloning.

setlocal

echo [setup] python check...
where python >nul 2>nul
if errorlevel 1 (
    echo [setup] python not on PATH. Install from python.org first.
    pause & exit /b 1
)

echo [setup] creating venv...
python -m venv venv

echo [setup] upgrading pip...
venv\Scripts\python.exe -m pip install --upgrade pip

echo [setup] installing CUDA torch + everything else (~3 GB)...
venv\Scripts\python.exe -m pip install ^
    --index-url https://download.pytorch.org/whl/cu128 ^
    --extra-index-url https://pypi.org/simple ^
    -r requirements.txt

echo [setup] verifying CUDA...
venv\Scripts\python.exe -c "import torch; print('cuda available:', torch.cuda.is_available(), 'device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"

echo.
echo [setup] DONE. Next: train the policy (one-shot):
echo   venv\Scripts\python.exe train_thief.py
endlocal
pause
