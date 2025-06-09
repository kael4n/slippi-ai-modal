@echo off
echo Starting repository cleanup...

REM Create directories
mkdir scripts\working 2>nul
mkdir scripts\archive 2>nul

echo Moving key working files...
REM Move your key files
move "scripts\experimental.py" "scripts\working\" 2>nul
move "scripts\slippi_WORKING.py" "scripts\working\" 2>nul
move "scripts\slippi_definitive_build(STABLE_WORKING).py" "scripts\working\stable_build.py" 2>nul
move "scripts\slippi_definitive_build(old_working).py" "scripts\working\old_build.py" 2>nul

echo Moving archived files...
REM Move version iterations to archive
move "scripts\patched_slippi_modal_strict_clean_v*.py" "scripts\archive\" 2>nul
move "scripts\slippi_fixed*.py" "scripts\archive\" 2>nul
move "scripts\slippi_numpy_fix*.py" "scripts\archive\" 2>nul
move "scripts\test*.py" "scripts\archive\" 2>nul
move "scripts\debug*.py" "scripts\archive\" 2>nul

echo Cleanup complete!
echo.
echo Next steps:
echo 1. Create .gitignore file
echo 2. Run: git rm -r --cached slp_parsed/
echo 3. Run: git add .
echo 4. Run: git commit -m "Reorganize project structure"
echo 5. Run: git push

pause