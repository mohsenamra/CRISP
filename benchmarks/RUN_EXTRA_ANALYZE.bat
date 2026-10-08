@echo off
cd /d "%~dp0"
set LOG=results\run_status_extra_analyze.txt
echo [%date% %time%] start > %LOG%
set PY=
where py >nul 2>nul && set PY=py -3
if not defined PY set PY=python
echo Analysing the supplementary experiments (no new solves, about 5-10 minutes)...
%PY% extra_analyze.py >> %LOG% 2>&1
if errorlevel 1 goto fail
echo [%date% %time%] ALL DONE >> %LOG%
echo ALL DONE. The analysis has finished.
pause
exit /b
:fail
echo [%date% %time%] FAILED >> %LOG%
echo An error occurred. Details are in results\run_status_extra_analyze.txt
pause
