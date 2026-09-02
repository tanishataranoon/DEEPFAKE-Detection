@echo off
cd /d F:\169\DEEPFAKE-Detection

echo ================================================================
echo AUTOMATED APPEARANCE EXPERIMENT
echo ================================================================
echo.
echo TRAINING STARTING...
echo.

call F:\169\.venv_cuda\Scripts\activate.bat

python -u training\Appreanch\150_appreanch_train.py

if errorlevel 1 (
    echo.
    echo ================================================================
    echo TRAINING FAILED - STOPPING
    echo ================================================================
    pause
    exit /b 1
)

echo.
echo ================================================================
echo TRAINING FINISHED SUCCESSFULLY
echo STARTING EVALUATION...
echo ================================================================
echo.

python -u evalution\appearanch_branch\150_appreanch_evaluate.py

if errorlevel 1 (
    echo.
    echo ================================================================
    echo EVALUATION FAILED - STOPPING
    echo ================================================================
    pause
    exit /b 1
)

echo.
echo ================================================================
echo EVALUATION FINISHED SUCCESSFULLY
echo STARTING VISUALIZATION...
echo ================================================================
echo.

python -u evalution\appearanch_branch\visualize_150_baseline.py

if errorlevel 1 (
    echo.
    echo ================================================================
    echo VISUALIZATION FAILED
    echo ================================================================
    pause
    exit /b 1
)

echo.
echo ================================================================
echo EVERYTHING FINISHED SUCCESSFULLY
echo ================================================================
echo.
pause