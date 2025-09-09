@echo off
REM Script to run Teams Gesture Camera with virtual environment
REM Change to the script's directory (where this bat file is located)
cd /d "%~dp0"

REM Activate the virtual environment
call .\venv\Scripts\activate.bat

REM Run the Teams Gesture Camera script
python teamsGestureCamera.py

REM Keep the window open if there's an error
if errorlevel 1 pause
