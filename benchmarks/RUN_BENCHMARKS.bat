@echo off
cd /d "%~dp0"
if not exist results mkdir results
set LOG=results\run_status.txt
echo [%date% %time%] start > %LOG%
set PY=
where py >nul 2>nul && set PY=py -3
if not defined PY where python >nul 2>nul && set PY=python
if not defined PY goto nopy
%PY% -c "import numpy,scipy,sklearn,highspy,torch; print('PACKAGES_OK')" >> %LOG% 2>&1
if errorlevel 1 goto fail
echo CRISP standard benchmarks: combinatorial auction, set cover, independent set, facility location.
echo About 4-6 hours (best run overnight). Keep the computer awake and plugged in.
echo You can close this window and re-run any time: finished parts are kept.
echo [%date% %time%] benchmarks >> %LOG%
%PY% bench_run.py 2>> %LOG%
if errorlevel 1 goto fail
echo [%date% %time%] analysis >> %LOG%
%PY% bench_analyze.py 2>> %LOG%
if errorlevel 1 goto fail
echo [%date% %time%] ALL DONE >> %LOG%
echo.
echo ALL DONE. The run has finished.
pause
exit /b
:fail
echo [%date% %time%] FAILED exit code %errorlevel% >> %LOG%
echo An error occurred. Details are in results\run_status.txt
pause
exit /b
:nopy
echo [%date% %time%] PYTHON_NOT_FOUND >> %LOG%
echo Python 3 was not found.
pause
