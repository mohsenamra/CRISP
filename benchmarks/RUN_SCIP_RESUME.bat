@echo off
cd /d "%~dp0"
set LOG=results\run_status_scip_resume.txt
echo [%date% %time%] start > %LOG%
set PY=
where py >nul 2>nul && set PY=py -3
if not defined PY set PY=python
echo Finishing the SCIP experiment on independent set (about 15-60 minutes), then re-running the analysis.
echo Each instance has a 15-minute safety limit, so the run cannot get stuck again.
%PY% scip_resume.py indset 2>> %LOG%
if errorlevel 1 goto fail
echo [%date% %time%] analysis >> %LOG%
%PY% extra_analyze.py >> %LOG% 2>&1
if errorlevel 1 goto fail
echo [%date% %time%] ALL DONE >> %LOG%
echo ALL DONE. The run has finished.
pause
exit /b
:fail
echo [%date% %time%] FAILED >> %LOG%
echo An error occurred. Details are in results\run_status_scip_resume.txt
pause
