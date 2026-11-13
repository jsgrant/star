@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "SRC=E:\Download(E_drv)\RTDicomStudio\RTDicomStudio\imported"
set "DST=%SRC%\FOV800 completed"
pip install -q pydicom numpy scipy
python fov_extend.py "%SRC%" "%DST%" --fov 800
pause
