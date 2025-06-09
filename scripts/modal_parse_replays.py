import modal
import sys
from pathlib import Path
import zipfile
import os

# --- Configuration ---
VOLUME_NAME = "slippi-ai-dataset-doesokay"
ZIP_FILE_NAME = "replays.zip"
EXTRACTED_REPLAYS_DIR = "/extracted_replays"
OUTPUT_DIR = "/parsed_output"
# --------------------

app = modal.App("slippi-parser-debugger")
volume = modal.NetworkFileSystem.from_name(VOLUME_NAME)

# --- THE DEFINITIVE FIX V2 ---
# We now use .run_commands() for pip install to ensure order of operations.
image = (
    modal.Image.debian_slim(python_version="3.9")
    # 1. Copy the necessary local files into the image's /root directory.
    .copy_local_file("requirements.txt", remote_path="/root/requirements.txt")
    .copy_local_dir("slippi_db", remote_path="/root/slippi_db")
    # 2. Install system packages.
    .apt_install("git")
    # 3. Run pip install using the file we know is now in the image.
    .run_commands("pip install -r /root/requirements.txt")
)
# -------------------------

@app.function(
    image=image,
    network_file_systems={
        "/root/data": volume
    },
    timeout=1800
)
def parse_replays():
    # Add /root to the path so Python can find the slippi_db module.
    sys.path.append("/root")
    try:
        from slippi_db import parse_local
        print("Successfully imported 'parse_local.py'.")
    except ImportError as e:
        print(f"FATAL: Could not import the parsing script. Error: {e}")
        return

    # The rest of the script remains the same
    zip_path = Path(f"/root/data/{ZIP_FILE_NAME}")
    extract_path = Path(f"/root/data{EXTRACTED_REPLAYS_DIR}")
    output_path = Path(f"/root/data{OUTPUT_DIR}")

    if not zip_path.exists():
        print(f"FATAL: The zip file was not found at {zip_path}. Please upload it first.")
        return

    print(f"Found zip file at {zip_path}. Starting extraction to {extract_path}...")
    extract_path.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_path)
    print("Extraction complete.")

    input_path = extract_path / '2024-10'
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Replay source directory: {input_path}")
    print(f"Output directory: {output_path}")

    if not any(input_path.glob('*.slp')):
        print(f"FATAL: No .slp files found in {input_path} after extraction.")
        return

    print(f"Found .slp files in {input_path}. Attempting to run the main parsing function...")
    try:
        class Args:
            root = input_path
            output = output_path
            threads = None
        
        parse_local.main(Args())
        print("--- Parsing function finished ---")
    except Exception as e:
        print(f"ERROR: The parsing function failed with an exception: {e}")
        import traceback
        traceback.print_exc()

    meta_file = output_path / 'meta.json'
    print(f"Verifying output... Checking for: {meta_file}")
    if meta_file.exists():
        print("SUCCESS! The 'meta.json' file was found.")
    else:
        print("FAILURE: The 'meta.json' file was NOT found after the script ran.")