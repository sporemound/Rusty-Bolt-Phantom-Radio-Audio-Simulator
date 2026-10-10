@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if not errorlevel 1 (
    py test_dsp_core.py
) else (
    python test_dsp_core.py
)
pause
endlocal
