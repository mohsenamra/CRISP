@echo off
cd /d "%~dp0"
if not exist results mkdir results
set LOG=results\run_status_ssoa.txt
echo [%date% %time%] start > %LOG%
set PY=
where py >nul 2>nul && set PY=py -3
if not defined PY set PY=python
echo CRISP application: supplier selection and order allocation built from real Codal data.
echo About 4-6 hours (best run overnight). Keep the computer awake and plugged in.
echo You can close this window and re-run any time: finished parts are kept.
echo Progress: results\progress_ssoa.txt
%PY% app_run.py 2>> %LOG%
if errorlevel 1 goto fail
echo [%date% %time%] ALL DONE >> %LOG%
echo ALL DONE. Tell Claude the run has finished.
pause
exit /b
:fail
echo [%date% %time%] FAILED >> %LOG%
echo An error occurred. Tell Claude: details are in results\run_status_ssoa.txt
pause
