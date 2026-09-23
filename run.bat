@echo off
REM ============================================================
REM  NeuralStream-Core · Windows Launcher
REM ============================================================

setlocal

REM Move to script directory
cd /d "%~dp0"

REM Default port (overridable with first argument)
set PORT=8501
if not "%~1"=="" set PORT=%~1

REM Default profile (overridable with second argument)
set PROFILE=default
if not "%~2"=="" set PROFILE=%~2

REM Force UTF-8 for emoji output
set PYTHONIOENCODING=utf-8
chcp 65001 >nul 2>&1

echo.
echo ============================================================
echo   NeuralStream-Core · Mission Control
echo ============================================================
echo   URL:     http://localhost:%PORT%
echo   Profile: %PROFILE%
echo   Stop:    Ctrl+C
echo ============================================================
echo.

REM Pass profile via env var so app.py picks it up
set NEURALSTREAM_PROFILE=%PROFILE%

REM Launch Streamlit
python -m streamlit run app.py --server.port %PORT%

endlocal