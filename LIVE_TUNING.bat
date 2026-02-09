@echo off
REM Live Eye Tracking Tuner
REM Runs tuning tool and animation simultaneously

cd /d "%~dp0"

REM Optional: set camera ID
REM set CAMERA_ID=1

REM Optional: customize animation window position/size
REM set ANIMATION_WIDTH=600
REM set ANIMATION_HEIGHT=500
REM set ANIMATION_X=1300
REM set ANIMATION_Y=100

echo ========================================
echo Live Eye Tracking Tuner
echo ========================================
echo.
echo This will open TWO windows:
echo   1. Tuning tool (left) - 7 OpenCV windows
echo   2. Animation (right) - Eye tracking display
echo.
echo Adjust parameters and see changes in REAL-TIME!
echo.
echo ========================================
echo.

py live_tuning.py

if errorlevel 1 (
    echo.
    echo Something went wrong! Check errors above.
    pause
)
