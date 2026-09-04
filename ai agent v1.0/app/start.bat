@echo off
setlocal

REM ============================================================
REM  MemFlow one-click launcher
REM  Fix: use %~dp0 (this script's own folder) instead of a
REM  hardcoded APP_DIR, so it never breaks when the project moves.
REM  NOTE: keep this file ASCII-only. Non-ASCII text breaks cmd.exe
REM  parsing under the default GBK codepage on Chinese Windows.
REM ============================================================

cd /d "%~dp0"

title MemFlow Launcher

REM Ollama executable: use PATH's "ollama" by default. If Ollama is installed in a
REM non-standard location, set the OLLAMA_EXE environment variable to its full path first.
if not defined OLLAMA_EXE set "OLLAMA_EXE=ollama"
set "HF_ENDPOINT=https://hf-mirror.com"

echo.
echo ==========================================
echo   MemFlow Local AI - One-Click Launcher
echo   Work dir: %cd%
echo ==========================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] python not found. Install Python and add it to PATH.
    echo.
    pause
    exit /b 1
)

echo [1/4] Checking Ollama service ...
netstat -ano | findstr ":11434" >nul 2>&1
if errorlevel 1 (
    echo       Starting Ollama ...
    if "%OLLAMA_EXE%"=="ollama" (
        where ollama >nul 2>&1
        if errorlevel 1 (
            echo       [ERROR] ollama not found in PATH. Install Ollama or set OLLAMA_EXE.
            echo.
            pause
            exit /b 1
        )
    ) else (
        if not exist "%OLLAMA_EXE%" (
            echo       [ERROR] %OLLAMA_EXE% not found. Install Ollama or set OLLAMA_EXE.
            echo.
            pause
            exit /b 1
        )
    )
    start "Ollama serve" /min "%OLLAMA_EXE%" serve
    timeout /t 5 /nobreak >nul
) else (
    echo       Ollama already running.
)

echo.
echo [2/4] Checking dependencies ...
pip show chromadb >nul 2>&1 && pip show sentence-transformers >nul 2>&1
if errorlevel 1 (
    echo       Installing requirements.txt ...
    pip install -r requirements.txt
) else (
    echo       OK.
)

echo.
echo [3/4] Migrating database ...
python migrate.py
if errorlevel 1 (
    echo       [WARN] migrate.py returned an error code, see output above.
)

echo.
echo [4/4] Starting Flask (port 5000) ...
set "FLASK_OK="
netstat -ano | findstr ":5000" >nul 2>&1
if errorlevel 1 (
    start "MemFlow Flask" cmd /k "set HF_ENDPOINT=%HF_ENDPOINT% && python app.py"
    timeout /t 4 /nobreak >nul
    netstat -ano | findstr ":5000" >nul 2>&1
    if errorlevel 1 (
        echo.
        echo       [WARN] Flask failed to start: port 5000 not listening.
        echo              Check the "MemFlow Flask" window for the traceback.
    ) else (
        echo       Flask started OK.
        set "FLASK_OK=1"
    )
) else (
    echo       Flask already running.
    set "FLASK_OK=1"
)

echo.
if defined FLASK_OK (
    start "" http://127.0.0.1:5000
    echo Done! Visit: http://127.0.0.1:5000
) else (
    echo Startup failed, browser not opened. Fix the error above and retry.
)
echo.
pause
