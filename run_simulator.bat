@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if not errorlevel 1 (
    py diode_effect_simulator.py
) else (
    python diode_effect_simulator.py
)
if errorlevel 1 pause
endlocal
