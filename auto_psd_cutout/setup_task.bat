@echo off
cd /d %~dp0

echo ============================================
echo   KoutuTools - ??????????
echo ============================================
echo.

set TASK_NAME=KoutuTools_Watch

REM ???????????
schtasks /delete /tn %TASK_NAME% /f 2>nul

REM ??????????????1????
schtasks /create ^
  /tn %TASK_NAME% ^
  /tr "cmd /c cd /d %CD% && python main.py --watch --no-photoshop" ^
  /sc onstart ^
  /ru SYSTEM ^
  /f

echo.
echo ?????????: schtasks /query /tn %TASK_NAME%
pause
