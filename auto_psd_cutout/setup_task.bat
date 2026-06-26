@echo off
cd /d %~dp0

echo ============================================
echo   KoutuTools - 注册开机自启监控任务
echo ============================================
echo.

set TASK_NAME=KoutuTools_Watch

REM 删除旧任务
schtasks /delete /tn %TASK_NAME% /f 2>nul

REM 注册新任务
schtasks /create ^
  /tn %TASK_NAME% ^
  /tr "cmd /c cd /d %CD% && python main.py --watch" ^
  /sc onlogon ^
  /f

if %errorlevel% equ 0 (
    echo.
    echo 注册成功！
    echo 手动启动: schtasks /run /tn %TASK_NAME%
    echo 停止运行: schtasks /end /tn %TASK_NAME%
    echo 删除任务: schtasks /delete /tn %TASK_NAME% /f
) else (
    echo.
    echo 注册失败，请以管理员身份运行。
)

pause
