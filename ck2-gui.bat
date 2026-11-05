@echo off
rem Run CorridorKey 2.0 from the source folder on Windows (developers).
rem Normal users install CK2-Setup.exe instead.
where uv >nul 2>nul
if errorlevel 1 (
  if exist "%LOCALAPPDATA%\ck2-gui\tools\uv.exe" (
    set "PATH=%LOCALAPPDATA%\ck2-gui\tools;%PATH%"
  ) else (
    echo uv is not installed. Get it from https://docs.astral.sh/uv/ or use CK2-Setup.exe.
    pause
    exit /b 1
  )
)
uv run --quiet --script "%~dp0ck2_gui.py" %*
