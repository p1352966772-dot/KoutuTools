@echo off
cd /d %~dp0
echo [1] 单次处理
echo [2] 监控模式(自动扫描新文件)
choice /c 12 /m "请选择
if errorlevel 2 goto watch
if errorlevel 1 goto once

:once
python main.py --no-photoshop
goto end

:watch
python main.py --watch --no-photoshop
goto end

:end
pause
