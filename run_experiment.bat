```bat
@echo off
cd /d E:\Depfake\DEEPFAKE-Detection

set "PYTHON=E:\Depfake\DEEPFAKE-Detection\.venv\Scripts\python.exe"

echo ================================================================
echo APPEARANCE BRANCH - FINAL EVALUATION ONLY
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

REM ================================================================
REM CREATE LOG DIRECTORY
REM ================================================================

if not exist "runs\appearance_logs" mkdir "runs\appearance_logs"

REM ================================================================
REM CREATE NEW TIMESTAMP FOR EVALUATION LOG
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

set "LOGFILE=runs\appearance_logs\appearance_evaluation_%D3%-%D2%-%D1%_%HH%-%MIN%.txt"

REM ================================================================
REM START NEW EVALUATION LOG
REM ================================================================

echo ================================================================ > "%LOGFILE%"
echo APPEARANCE BRANCH - FINAL TEST EVALUATION >> "%LOGFILE%"
echo ================================================================ >> "%LOGFILE%"
echo Project    : E:\Depfake\DEEPFAKE-Detection >> "%LOGFILE%"
echo Started    : %date% %time% >> "%LOGFILE%"
echo Python     : %PYTHON% >> "%LOGFILE%"
echo Checkpoint : checkpoints\appearance_full\best.pt >> "%LOGFILE%"
echo Test CSV   : data\split\test.csv >> "%LOGFILE%"
echo ================================================================ >> "%LOGFILE%"
echo. >> "%LOGFILE%"

echo New evaluation log:
echo %LOGFILE%
echo.

REM ================================================================
REM GPU CHECK
REM ================================================================

echo ================================================================
echo CHECKING GPU
echo ================================================================
echo.

"%PYTHON%" -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')" 2>&1 | powershell -Command "$input | Tee-Object -FilePath '%LOGFILE%' -Append"

if errorlevel 1 (
    echo.
    echo GPU CHECK FAILED.
    echo GPU CHECK FAILED: %date% %time% >> "%LOGFILE%"
    pause
    exit /b 1
)

echo.

REM ================================================================
REM CHECK EXISTING CHECKPOINT
REM ================================================================

echo ================================================================
echo CHECKING EXISTING CHECKPOINT
echo ================================================================
echo.

if not exist "checkpoints\appearance_full\best.pt" (
    echo ERROR: best.pt was not found.
    echo Expected:
    echo checkpoints\appearance_full\best.pt

    echo CHECKPOINT NOT FOUND: %date% %time% >> "%LOGFILE%"
    pause
    exit /b 1
)

echo Existing best.pt found.
echo.

echo Checkpoint found successfully. >> "%LOGFILE%"
echo. >> "%LOGFILE%"

REM ================================================================
REM FINAL TEST EVALUATION
REM ================================================================

echo ================================================================
echo STARTING FINAL TEST EVALUATION
echo ================================================================
echo.
echo NO TRAINING WILL BE PERFORMED.
echo Existing best.pt will be evaluated on the held-out test set.
echo.

echo ================================================================ >> "%LOGFILE%"
echo FINAL TEST EVALUATION STARTED: %date% %time% >> "%LOGFILE%"
echo ================================================================ >> "%LOGFILE%"
echo NO TRAINING PERFORMED. >> "%LOGFILE%"
echo. >> "%LOGFILE%"

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
echo EVALUATION FINISHED SUCCESSFULLY: %date% %time% >> "%LOGFILE%"
echo ================================================================ >> "%LOGFILE%"
echo. >> "%LOGFILE%"

REM ================================================================
REM VERIFY TEST RESULTS
REM ================================================================

if not exist "checkpoints\appearance_full\test_results.json" (
    echo.
    echo ERROR: test_results.json was not created.
    echo.

    echo ERROR: test_results.json was not created: %date% %time% >> "%LOGFILE%"
    pause
    exit /b 1
)

echo test_results.json created successfully.
echo.

echo Results file: checkpoints\appearance_full\test_results.json >> "%LOGFILE%"
echo. >> "%LOGFILE%"

REM ================================================================
REM VISUALIZATION
REM ================================================================

echo ================================================================
echo STARTING VISUALIZATION
echo ================================================================
echo.

echo ================================================================ >> "%LOGFILE%"
echo VISUALIZATION STARTED: %date% %time% >> "%LOGFILE%"
echo ================================================================ >> "%LOGFILE%"

"%PYTHON%" -u evalution\appearanch_branch\visualize_full_dataset.py 2>&1 | powershell -Command "$input | Tee-Object -FilePath '%LOGFILE%' -Append"

if errorlevel 1 (
    echo.
    echo ================================================================
    echo VISUALIZATION FAILED - STOPPING
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
echo VISUALIZATION FINISHED SUCCESSFULLY: %date% %time% >> "%LOGFILE%"
echo ================================================================ >> "%LOGFILE%"
echo. >> "%LOGFILE%"

REM ================================================================
REM COMPLETE
REM ================================================================

echo ================================================================
echo EVALUATION + VISUALIZATION COMPLETED
echo ================================================================
echo.
echo TRAINING WAS NOT RUN.
echo.
echo Checkpoint:
echo checkpoints\appearance_full\best.pt
echo.
echo Test results:
echo checkpoints\appearance_full\test_results.json
echo.
echo New evaluation log:
echo %LOGFILE%
echo.

echo ================================================================ >> "%LOGFILE%"
echo FINAL EVALUATION COMPLETED SUCCESSFULLY >> "%LOGFILE%"
echo ================================================================ >> "%LOGFILE%"
echo Training was NOT performed. >> "%LOGFILE%"
echo Checkpoint: checkpoints\appearance_full\best.pt >> "%LOGFILE%"
echo Test results: checkpoints\appearance_full\test_results.json >> "%LOGFILE%"
echo Finished: %date% %time% >> "%LOGFILE%"
echo Complete log: %LOGFILE% >> "%LOGFILE%"
echo ================================================================ >> "%LOGFILE%"

pause
```
