@echo off
cd /d %~dp0

echo ============================================
echo   KoutuTools - Setup
echo ============================================
echo.

echo [1/3] Installing VC++ Redist...
winget install Microsoft.VCRedist.2015+.x64 --accept-package-agreements --silent 2>nul
if %errorlevel% neq 0 (
    echo Please install VC++ Redist manually:
    echo https://aka.ms/vs/17/release/vc_redist.x64.exe
)

echo.
echo [2/3] Installing Python packages...
pip install -r requirements.txt

echo.
echo [3/3] Creating folders...
if not exist input mkdir input
if not exist output mkdir output

echo.
echo ============================================
echo   Setup Complete!
echo   Run: python main.py --watch
echo ============================================
pause
