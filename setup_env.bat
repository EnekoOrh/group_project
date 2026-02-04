@echo off
set "VENV_DIR=.venv"

echo ==========================================
echo     Project Setup Script (Windows)
echo ==========================================

:: Check if python is available
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: 'python' is not installed or not in PATH.
    pause
    exit /b 1
)

:: Create virtual environment if it doesn't exist
if not exist "%VENV_DIR%" (
    echo Creating virtual environment in %VENV_DIR%...
    python -m venv "%VENV_DIR%"
) else (
    echo Virtual environment already exists in %VENV_DIR%.
)

:: Activate the virtual environment
echo Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"

:: Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

:: Install dependencies
if exist "requirements.txt" (
    echo Installing dependencies from requirements.txt...
    pip install -r requirements.txt
) else (
    echo Warning: requirements.txt not found!
)

echo ==========================================
echo           Setup Complete!
echo ==========================================
echo.
echo To activate the environment manually, run:
echo     %VENV_DIR%\Scripts\activate
echo.
pause
