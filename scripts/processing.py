import modal
import os
import json
import subprocess
from pathlib import Path

# Same setup as your training script
DATASET_VOLUME_NAME = "slippi-ai-dataset-doesokay"
PROJECT_ROOT = "/root/slippi-ai"
REPO_URL = "https://github.com/vladfi1/slippi-ai.git"
PEPPI_PY_COMMIT_URL = "git+https://github.com/hohav/peppi-py.git@8c02a4659c3302321dfbfcf2093c62f634e335f7"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "build-essential", "pkg-config", "libssl-dev", "curl")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
    )
    .env({
        "PATH": "/root/.cargo/bin:$PATH",
        "TF_ENABLE_ONEDNN_OPTS": "0",
    })
    .run_commands(
        f"git clone {REPO_URL} {PROJECT_ROOT}",
    )
    .workdir(PROJECT_ROOT)
    .run_commands(
        ". ~/.cargo/env && pip install --upgrade pip setuptools wheel",
        ". ~/.cargo/env && pip install maturin",
        f". ~/.cargo/env && pip install --no-build-isolation '{PEPPI_PY_COMMIT_URL}'",
        "pip install -r requirements.txt",
        "pip install -e .",
    )
)

app = modal.App("slippi-data-fixer")
dataset_volume = modal.Volume.from_name(DATASET_VOLUME_NAME)

@app.function(
    image=image,
    volumes={"/dataset": dataset_volume},
    timeout=7200,
)
def reprocess_data():
    """Reprocess the slippi data with the correct format."""
    os.chdir(PROJECT_ROOT)
    
    print("=== Reprocessing Slippi Data ===")
    
    # Check if we have raw .slp files to reprocess
    dataset_path = Path("/dataset")
    slp_files = list(dataset_path.glob("**/*.slp"))
    
    if not slp_files:
        print("❌ No .slp files found in /dataset")
        print("Please upload your .slp replay files to the dataset volume")
        return
    
    print(f"Found {len(slp_files)} .slp files")
    
    # Create a temporary directory for raw replays
    raw_dir = dataset_path / "raw"
    raw_dir.mkdir(exist_ok=True)
    
    # Move .slp files to raw directory if they're not already there
    if not (raw_dir / "*.slp").exists():
        for slp_file in slp_files:
            if slp_file.parent != raw_dir:
                print(f"Moving {slp_file.name} to raw directory")
                slp_file.rename(raw_dir / slp_file.name)
    
    # Clear any existing processed data
    parsed_dir = dataset_path / "Parsed"
    if parsed_dir.exists():
        print("Removing existing Parsed directory")
        import shutil
        shutil.rmtree(parsed_dir)
    
    meta_file = dataset_path / "meta.json"
    if meta_file.exists():
        print("Removing existing meta.json")
        meta_file.unlink()
    
    # Run the preprocessing script
    print("Running slippi_db/parse_local.py...")
    
    try:
        import sys
        sys.path.append(PROJECT_ROOT)
        
        # Import and run the parser directly
        from slippi_db import parse_local
        
        # Set up arguments for the parser
        sys.argv = [
            'parse_local.py',
            '--input_dir', str(raw_dir),
            '--output_dir', str(dataset_path),
            '--num_workers', '4',
            '--include_netplay', 'True',
            '--include_locals', 'True'
        ]
        
        print(f"Running parser with args: {sys.argv}")
        parse_local.main()
        
    except Exception as e:
        print(f"❌ Error running parse_local.py: {e}")
        import traceback
        traceback.print_exc()
        
        # Try alternative approach with subprocess
        print("Trying alternative subprocess approach...")
        try:
            cmd = [
                'python', '-m', 'slippi_db.parse_local',
                '--input_dir', str(raw_dir),
                '--output_dir', str(dataset_path),
                '--num_workers', '4',
                '--include_netplay', 'True',
                '--include_locals', 'True'
            ]
            
            result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
            print(f"Command output: {result.stdout}")
            if result.stderr:
                print(f"Command errors: {result.stderr}")
            
            if result.returncode != 0:
                raise RuntimeError(f"Parser failed with return code {result.returncode}")
                
        except Exception as e2:
            print(f"❌ Subprocess approach also failed: {e2}")
            return
    
    # Verify the output
    print("=== Verifying Processed Data ===")
    
    if not parsed_dir.exists():
        print("❌ Parsed directory was not created")
        return
    
    if not meta_file.exists():
        print("❌ meta.json was not created")
        return
    
    parsed_files = list(parsed_dir.glob("*.json"))
    print(f"✅ Created {len(parsed_files)} parsed files")
    
    # Check meta.json structure
    try:
        with open(meta_file, 'r') as f:
            meta_content = json.load(f)
        print(f"✅ meta.json loaded successfully with {len(meta_content)} entries")
        
        # Check first entry structure
        if meta_content:
            first_key = list(meta_content.keys())[0]
            first_entry = meta_content[first_key]
            print(f"First entry structure: {type(first_entry)}")
            if isinstance(first_entry, dict):
                print(f"First entry keys: {list(first_entry.keys())}")
    except Exception as e:
        print(f"❌ Error verifying meta.json: {e}")
    
    print("=== Data Reprocessing Complete ===")

@app.local_entrypoint()
def main():
    print("Starting data reprocessing...")
    reprocess_data.remote()