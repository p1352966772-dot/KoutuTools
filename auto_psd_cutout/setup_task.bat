@echo off
cd /d %~dp0

echo ============================================
echo   KoutuTools - Register Auto-Start Task
echo ============================================
echo.

set TASK_NAME=KoutuTools_Watch

schtasks /delete /tn %TASK_NAME% /f 2>nul

schtasks /create ^
  /tn %TASK_NAME% ^
  /tr "cmd /c cd /d %CD% && python main.py --watch" ^
  /sc onlogon ^
  /f

if %errorlevel% equ 0 (
    echo.
    echo SUCCESS - Task registered
    echo Start : schtasks /run   /tn %TASK_NAME%
    echo Stop  : schtasks /end   /tn %TASK_NAME%
    echo Delete: schtasks /delete /tn %TASK_NAME% /f
) else (
    echo.
    echo FAILED - Run as Administrator
)

pause
