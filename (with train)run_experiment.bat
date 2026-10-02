@echo off
cd /d E:\Depfake\DEEPFAKE-Detection

set "PYTHON=E:\Depfake\DEEPFAKE-Detection\.venv\Scripts\python.exe"

echo ================================================================
echo AUTOMATED APPEARANCE EXPERIMENT - FULL DATASET
echo ================================================================
echo.
echo PROJECT : E:\Depfake\DEEPFAKE-Detection
echo PYTHON  : %PYTHON%
echo.

if not exist "%PYTHON%" (
echo ERROR: Python executable not found.
echo %PYTHON%
pause
exit /b 1
)

echo Python executable found.
echo.

REM ================================================================
REM CREATE LOG DIRECTORY
REM ================================================================

if not exist "runs\appearance_logs" mkdir "runs\appearance_logs"

REM ================================================================
REM CREATE TIMESTAMP
REM ================================================================

for /f "tokens=1-3 delims=/ " %%a in ("%date%") do (
set "D1=%%a"
set "D2=%%b"
set "D3=%%c"
)

for /f "tokens=1-2 delims=:." %%a in ("%time%") do (
set "HH=%%a"
set "MIN=%%b"
)

set "HH=%HH: =0%"

set "LOGFILE=runs\appearance_logs\appearance_experiment_%D3%-%D2%-%D1%_%HH%-%MIN%.txt"

echo Log file:
echo %LOGFILE%
echo.

echo ================================================================ > "%LOGFILE%"
echo APPEARANCE BRANCH - FULL DATASET EXPERIMENT >> "%LOGFILE%"
echo ================================================================ >> "%LOGFILE%"
echo Project : E:\Depfake\DEEPFAKE-Detection >> "%LOGFILE%"
echo Started : %date% %time% >> "%LOGFILE%"
echo Python  : %PYTHON% >> "%LOGFILE%"
echo ================================================================ >> "%LOGFILE%"
echo. >> "%LOGFILE%"

REM ================================================================
REM GPU CHECK
REM ================================================================

echo ================================================================
echo CHECKING GPU
echo ================================================================
echo.

"%PYTHON%" -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')" 2>&1 | powershell -Command "$input | Tee-Object -FilePath '%LOGFILE%' -Append"

if errorlevel 1 (
echo GPU CHECK FAILED.
pause
exit /b 1
)

echo.

REM ================================================================
REM TRAINING
REM ================================================================

echo ================================================================
echo TRAINING STARTING
echo ================================================================
echo.

"%PYTHON%" -u training\Appreanch\train_whole_dataset.py 2>&1 | powershell -Command "$input | Tee-Object -FilePath '%LOGFILE%' -Append"

if errorlevel 1 (
echo.
echo ================================================================
echo TRAINING FAILED - STOPPING
echo ================================================================
echo.
echo TRAINING FAILED: %date% %time% >> "%LOGFILE%"
pause
exit /b 1
)

echo.
echo ================================================================
echo TRAINING FINISHED SUCCESSFULLY
echo ================================================================
echo.

echo ================================================================ >> "%LOGFILE%"
echo TRAINING FINISHED: %date% %time% >> "%LOGFILE%"
echo ================================================================ >> "%LOGFILE%"
echo. >> "%LOGFILE%"

REM ================================================================
REM EVALUATION
REM ================================================================

echo ================================================================
echo STARTING EVALUATION
echo ================================================================
echo.

"%PYTHON%" -u evalution\appearanch_branch\appearance_evaluation.py 2>&1 | powershell -Command "$input | Tee-Object -FilePath '%LOGFILE%' -Append"

if errorlevel 1 (
echo.
echo ================================================================
echo EVALUATION FAILED - STOPPING
echo ================================================================
echo.
echo EVALUATION FAILED: %date% %time% >> "%LOGFILE%"
pause
exit /b 1
)

echo.
echo ================================================================
echo EVALUATION FINISHED SUCCESSFULLY
echo ================================================================
echo.

echo ================================================================ >> "%LOGFILE%"
echo EVALUATION FINISHED: %date% %time% >> "%LOGFILE%"
echo ================================================================ >> "%LOGFILE%"
echo. >> "%LOGFILE%"

REM ================================================================
REM VISUALIZATION
REM ================================================================

echo ================================================================
echo STARTING VISUALIZATION
echo ================================================================
echo.

"%PYTHON%" -u evalution\appearanch_branch\visualize_full_dataset.py 2>&1 | powershell -Command "$input | Tee-Object -FilePath '%LOGFILE%' -Append"

if errorlevel 1 (
echo.
echo ================================================================
echo VISUALIZATION FAILED
echo ================================================================
echo.
echo VISUALIZATION FAILED: %date% %time% >> "%LOGFILE%"
pause
exit /b 1
)

echo.
echo ================================================================
echo VISUALIZATION FINISHED SUCCESSFULLY
echo ================================================================
echo.

echo ================================================================ >> "%LOGFILE%"
echo VISUALIZATION FINISHED: %date% %time% >> "%LOGFILE%"
echo ================================================================ >> "%LOGFILE%"
echo. >> "%LOGFILE%"

REM ================================================================
REM COMPLETE
REM ================================================================

echo ================================================================
echo EVERYTHING FINISHED SUCCESSFULLY
echo ================================================================
echo.
echo COMPLETE LOG:
echo %LOGFILE%
echo.

echo ================================================================ >> "%LOGFILE%"
echo EVERYTHING FINISHED SUCCESSFULLY >> "%LOGFILE%"
echo Finished: %date% %time% >> "%LOGFILE%"
echo ================================================================ >> "%LOGFILE%"

pause
