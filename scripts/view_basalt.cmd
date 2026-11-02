@echo off
rem Watch a recording in Basalt's own window (camera tracks, map points, trajectory), at about real time.
rem   scripts\view_basalt.cmd data\slam\rec_20261026_155811
rem Needs D:\slam\basalt built (see docs\slam-record-guide.md). Close the window to quit.
if "%~1"=="" (echo usage: scripts\view_basalt.cmd ^<recording folder^> [ms per frame, default 75] & exit /b 1)
set REC=%~f1
set MS=%2
if "%MS%"=="" set MS=75
set PATH=D:\slam\vcpkg\installed\x64-windows\bin;%PATH%
cd /d D:\slam\basalt
.\build\basalt_vio.exe --show-gui 1 --playback-ms %MS% --dataset-path "%REC%" --dataset-type euroc --cam-calib "%REC%\calib.json" --config-path data\euroc\euroc_config.json
