@echo off
REM HQDE v0.1.5 Publishing Script for Windows

echo 🚀 Publishing HQDE v0.1.5 to PyPI
echo ==================================
echo.

REM Step 1: Clean previous builds
echo 📦 Step 1: Cleaning previous builds...
if exist dist rmdir /s /q dist
if exist build rmdir /s /q build
if exist hqde.egg-info rmdir /s /q hqde.egg-info
echo ✅ Cleaned
echo.

REM Step 2: Build the package
echo 🔨 Step 2: Building package...
python -m build
if errorlevel 1 (
    echo ❌ Build failed!
    exit /b 1
)
echo ✅ Built successfully
echo.

REM Step 3: Check the distribution
echo 🔍 Step 3: Checking distribution...
python -m twine check dist/*
if errorlevel 1 (
    echo ❌ Distribution check failed!
    exit /b 1
)
echo ✅ Distribution is valid
echo.

REM Step 4: Upload to PyPI
echo 📤 Step 4: Uploading to PyPI...
echo You will be prompted for your PyPI credentials
python -m twine upload dist/*
if errorlevel 1 (
    echo ❌ Upload failed!
    exit /b 1
)
echo ✅ Uploaded successfully
echo.

REM Step 5: Verify
echo ✅ Step 5: Verifying installation...
echo Run this command to test:
echo   pip install hqde==0.1.5 --upgrade
echo   python -c "import hqde; print(hqde.__version__)"
echo.

echo 🎉 HQDE v0.1.5 published successfully!
echo.
echo 📝 Next steps:
echo 1. Update your Kaggle notebook: !pip install hqde==0.1.5 --upgrade
echo 2. Verify version: import hqde; print(hqde.__version__)
echo 3. Run training with 40 epochs
echo 4. Look for 'Weights aggregated and synchronized' messages
echo.
echo Expected improvements:
echo   - MNIST: ~99.2%% (from ~98%%)
echo   - Fashion-MNIST: ~91-92%% (from ~87%%)
echo   - CIFAR-10: ~75-80%% (from ~59%%)
echo   - SVHN: ~85-88%% (from ~72%%)
echo   - CIFAR-100: ~45-55%% (from ~14%%)

pause
