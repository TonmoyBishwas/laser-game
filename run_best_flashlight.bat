@echo off
REM ============================================================================
REM  KNOWN-GOOD flashlight baseline (the smoothest config so far).
REM
REM  This is the fallback you asked to preserve: the PyTorch .pt model at
REM  imgsz 320, inference UNCAPPED, render capped at 30 fps. On the uni i7 this
REM  gave ~9-10 fps inference and the smoothest tracking before we tried ONNX.
REM
REM  Use this whenever you want to get back to the best-feeling setup.
REM  (NOT ONNX, NOT OpenVINO — plain PyTorch .pt, which felt best.)
REM ============================================================================

venv\Scripts\python.exe flashlight_sandbox.py --imgsz 320 --max-detect-fps 0 --render-fps 30

pause
