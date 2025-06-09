# Repository Cleanup Commands

# 1. Create the organized directory structure
mkdir -p scripts/archive
mkdir -p scripts/working
mkdir -p scripts/deprecated

# 2. Keep your main working files in scripts/working/
move scripts/experimental.py scripts/working/
move "scripts/slippi_WORKING.py" scripts/working/
move "scripts/slippi_definitive_build(STABLE_WORKING).py" scripts/working/stable_build.py
move "scripts/slippi_definitive_build(old_working).py" scripts/working/old_build.py

# 3. Move all the version iterations to archive
move scripts/patched_slippi_modal_strict_clean_v*.py scripts/archive/
move scripts/slippi_fixed*.py scripts/archive/
move scripts/slippi_numpy_fix*.py scripts/archive/
move scripts/test*.py scripts/archive/
move scripts/debug*.py scripts/archive/

# 4. Keep useful utilities in main scripts/
# (Keep train_on_modal.py, upload_to_modal.py, prepare_replays.py)

# 5. Add to .gitignore (put this in your .gitignore file)
echo "# Archive directory - old experimental files" >> .gitignore
echo "scripts/archive/" >> .gitignore

# 6. Remove parsed data from git tracking
git rm -r --cached slp_parsed/

# 7. Add and commit the cleanup
git add .gitignore
git add scripts/
git commit -m "Reorganize project structure and add .gitignore"