@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
rem ---------------------------------------------------------------
rem  CT FOV 800 변환 (fov_extend.py)
rem  사용: 이 파일 위에 환자 폴더를 끌어다 놓거나, 더블클릭 후 폴더 선택
rem  결과: <고른 폴더>\FOV800 completed\<환자>\<시리즈>\
rem ---------------------------------------------------------------
where python >nul 2>nul
if errorlevel 1 (
  echo [오류] Python 이 없습니다. python.org 에서 Python 3.12 를 설치하세요 ^(Add python.exe to PATH 체크^).
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  echo 처음 실행: 필요한 패키지를 이 폴더의 .venv 에 설치합니다...
  python -m venv .venv || (echo [오류] 가상환경을 만들지 못했습니다. & pause & exit /b 1)
  ".venv\Scripts\python.exe" -m pip install -q --upgrade pip
  ".venv\Scripts\python.exe" -m pip install -q -r requirements.txt || (echo [오류] 패키지 설치 실패 ^(인터넷 확인^). & pause & exit /b 1)
)
set "SRC=%~1"
if "%SRC%"=="" (
  for /f "usebackq delims=" %%I in (`powershell -NoProfile -STA -ExecutionPolicy Bypass -File "%~dp0pick_folder.ps1"`) do set "SRC=%%I"
)
if "%SRC%"=="" (
  echo 취소했습니다.
  pause
  exit /b 1
)
echo.
echo 입력 폴더: %SRC%
echo 결과 폴더: %SRC%\FOV800 completed
echo.
".venv\Scripts\python.exe" fov_extend.py "%SRC%" --fov 800
echo.
pause
