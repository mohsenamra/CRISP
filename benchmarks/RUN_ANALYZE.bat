@echo off
cd /d "%~dp0"
set PY=
where py >nul 2>nul && set PY=py -3
if not defined PY where python >nul 2>nul && set PY=python
%PY% bench_analyze.py
echo.
echo ALL DONE. Tell Claude the analysis has finished.
pause
