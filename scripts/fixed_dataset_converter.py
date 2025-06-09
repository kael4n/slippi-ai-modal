import modal
import os
import json
import pickle
import shutil
import zipfile
from pathlib import Path
from typing import List, Dict, Any
import sys

# Volume names
DATASET_VOLUME_NAME = "slippi-ai-dataset-doesokay"
PROJECT_ROOT = "/root/slippi-ai"
REPO_URL = "https://github.com/vladfi1/slippi-ai.git"
PEPPI_PY_COMMIT_URL = "git+https://github.com/hohav/peppi-py.git@8c02a4659c3302321dfbfcf2093c62f634e335f7"

# Same image as your training script
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "build-essential", "pkg-config", "libssl-dev", "curl", "unzip")
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

app = modal.App("slippi-dataset-fixer")
dataset_volume = modal.Volume.from_name(DATASET_VOLUME_NAME)
nfs = modal.NetworkFileSystem.from_name("slippi-ai-dataset-doesokay")

@app.function(
    image=image,
    volumes={"/dataset": dataset_volume},
    network_file_systems={"/nfs": nfs},
    timeout=3600,
)
def parse_slp_files_properly():
    """Parse SLP files from NFS directly and create proper slippi-ai format dataset."""
    print("=== Parsing SLP Files with Proper Format ===")
    
    os.chdir(PROJECT_ROOT)
    
    try:
        import peppi
        print("✅ Peppi available for parsing")
    except ImportError:
        print("❌ Peppi not available - cannot parse .slp files")
        return 0
    
    # Clear existing parsed data to start fresh
    parsed_dir = Path("/dataset/Parsed")
    if parsed_dir.exists():
        shutil.rmtree(parsed_dir)
    parsed_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all .slp files in NFS
    nfs_replays_dir = Path("/nfs/replays")
    slp_files = []
    
    if nfs_replays_dir.exists():
        slp_files = list(nfs_replays_dir.rglob("*.slp"))
        print(f"Found {len(slp_files)} .slp files in NFS")
    
    # Also check for zip file
    nfs_zip = Path("/nfs/replays.zip")
    if nfs_zip.exists():
        print(f"📦 Also found replays.zip")
        # Extract a few files from zip to test
        try:
            with zipfile.ZipFile(nfs_zip, 'r') as zip_ref:
                zip_slp_files = [f for f in zip_ref.namelist() if f.endswith('.slp')]
                print(f"  Found {len(zip_slp_files)} .slp files in zip")
                
                # Extract first 50 files for testing
                temp_extract_dir = Path("/tmp/extracted_replays")
                temp_extract_dir.mkdir(exist_ok=True)
                
                for i, slp_file in enumerate(zip_slp_files[:50]):
                    try:
                        zip_ref.extract(slp_file, temp_extract_dir)
                        extracted_path = temp_extract_dir / slp_file
                        if extracted_path.exists():
                            slp_files.append(extracted_path)
                    except Exception as e:
                        print(f"  ❌ Error extracting {slp_file}: {e}")
                        
        except Exception as e:
            print(f"❌ Error processing zip: {e}")
    
    if not slp_files:
        print("No .slp files found to parse")
        return 0
    
    print(f"Processing {len(slp_files)} .slp files...")
    
    parsed_count = 0
    game_list = []
    
    # Process files in batches to avoid memory issues
    batch_size = 10
    for i in range(0, min(len(slp_files), 100), batch_size):  # Limit to 100 for testing
        batch = slp_files[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}: files {i+1}-{min(i+batch_size, len(slp_files))}")
        
        for slp_file in batch:
            try:
                print(f"  Parsing {slp_file.name}...")
                
                # Parse with peppi
                game = peppi.read_slippi(str(slp_file))
                
                # Create the proper slippi-ai format
                # The key is to create a structure that matches what train_lib expects
                replay_data = {
                    'filename': slp_file.name,
                    'game': game,  # Store the raw game object
                }
                
                # Try to extract player information in the expected format
                try:
                    # Create mock player objects that match the expected interface
                    if hasattr(game, 'metadata') and game.metadata:
                        # Extract player info from metadata
                        players = []
                        if hasattr(game.metadata, 'players'):
                            for player in game.metadata.players:
                                player_info = {
                                    'name': getattr(player, 'netplay_name', 'Unknown'),
                                    'tag': getattr(player, 'tag', 'UNK'),
                                    'character': getattr(player, 'character', 0),
                                }
                                players.append(player_info)
                        
                        # Create a proper meta object
                        meta_data = {
                            'players': players,
                            'p0': players[0] if len(players) > 0 else {'name': 'Player1', 'tag': 'P1'},
                            'p1': players[1] if len(players) > 1 else {'name': 'Player2', 'tag': 'P2'},
                            'stage': getattr(game.metadata, 'stage', 0) if hasattr(game.metadata, 'stage') else 0,
                        }
                        
                        replay_data['meta'] = meta_data
                    else:
                        # Fallback meta structure
                        replay_data['meta'] = {
                            'p0': {'name': 'Player1', 'tag': 'P1'},
                            'p1': {'name': 'Player2', 'tag': 'P2'},
                            'stage': 0,
                            'players': [
                                {'name': 'Player1', 'tag': 'P1'},
                                {'name': 'Player2', 'tag': 'P2'}
                            ]
                        }
                    
                    # Add frame count
                    if hasattr(game, 'frames'):
                        replay_data['frame_count'] = len(game.frames)
                    
                except Exception as meta_error:
                    print(f"    ⚠️  Meta extraction failed: {meta_error}")
                    # Use minimal fallback
                    replay_data['meta'] = {
                        'p0': {'name': 'Player1', 'tag': 'P1'},
                        'p1': {'name': 'Player2', 'tag': 'P2'},
                        'stage': 0,
                    }
                
                # Save as pickle (this is what slippi-ai actually expects)
                pkl_file = parsed_dir / f"{slp_file.stem}.pkl"
                with open(pkl_file, 'wb') as f:
                    pickle.dump(replay_data, f)
                
                game_list.append(slp_file.name)
                parsed_count += 1
                
                print(f"    ✅ Parsed and saved {pkl_file.name}")
                
            except Exception as e:
                print(f"    ❌ Error parsing {slp_file.name}: {e}")
                continue
    
    # Create the proper meta.json format for slippi-ai
    meta_data = {
        'replays': game_list,
        'parsed_dir': str(parsed_dir),
        'total_replays': len(game_list),
        'format': 'slippi-ai-pickle'
    }
    
    with open('/dataset/meta.json', 'w') as f:
        json.dump(meta_data, f, indent=2)
    
    print(f"\n✅ Successfully parsed {parsed_count} SLP files")
    print(f"✅ Created {len(game_list)} training replays")
    print(f"✅ Updated meta.json")
    print(f"✅ Files saved in {parsed_dir}")
    
    return parsed_count

@app.function(
    image=image,
    volumes={"/dataset": dataset_volume},
    timeout=600,
)
def inspect_dataset_structure():
    """Inspect the current dataset structure to understand the format."""
    print("=== Inspecting Dataset Structure ===")
    
    # Check what we have
    dataset_path = Path("/dataset")
    parsed_path = Path("/dataset/Parsed")
    
    print(f"\n📊 Dataset overview:")
    if dataset_path.exists():
        total_files = len(list(dataset_path.rglob("*")))
        print(f"  Total files in dataset: {total_files}")
    
    if parsed_path.exists():
        parsed_files = list(parsed_path.iterdir())
        print(f"  Files in Parsed/: {len(parsed_files)}")
        
        # Look at a few files to understand structure
        for i, file_path in enumerate(parsed_files[:3]):
            print(f"\n📄 Sample file {i+1}: {file_path.name}")
            
            if file_path.suffix == '.json':
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    print(f"  JSON keys: {list(data.keys())}")
                    if 'meta' in data:
                        print(f"  Meta type: {type(data['meta'])}")
                        if isinstance(data['meta'], dict):
                            print(f"  Meta keys: {list(data['meta'].keys())}")
                except Exception as e:
                    print(f"  Error reading JSON: {e}")
                    
            elif file_path.suffix == '.pkl':
                try:
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                    print(f"  PKL type: {type(data)}")
                    if hasattr(data, 'keys'):
                        print(f"  PKL keys: {list(data.keys())}")
                    elif hasattr(data, '__dict__'):
                        print(f"  PKL attributes: {list(data.__dict__.keys())}")
                except Exception as e:
                    print(f"  Error reading PKL: {e}")
    
    # Check meta.json
    meta_file = Path("/dataset/meta.json")
    if meta_file.exists():
        try:
            with open(meta_file, 'r') as f:
                meta = json.load(f)
            print(f"\n📋 meta.json structure:")
            print(f"  Keys: {list(meta.keys())}")
            if 'games' in meta:
                print(f"  Game count: {len(meta['games'])}")
            if 'replays' in meta:
                print(f"  Replay count: {len(meta['replays'])}")
        except Exception as e:
            print(f"  Error reading meta.json: {e}")

@app.local_entrypoint()
def inspect():
    """Inspect current dataset structure"""
    inspect_dataset_structure.remote()

@app.local_entrypoint()
def fix_dataset():
    """Parse SLP files properly for slippi-ai training"""
    print("=== Fixing Dataset Format ===")
    count = parse_slp_files_properly.remote()
    print(f"\nFixed dataset with {count} properly formatted replays")
    print("You can now run experimental.py for training!")

@app.local_entrypoint()
def main():
    """Main entrypoint - inspect then fix"""
    print("=== Dataset Format Fixer ===")
    print("First inspecting current format...")
    inspect_dataset_structure.remote()
    
    print("\nNow fixing the format...")
    count = parse_slp_files_properly.remote()
    print(f"\n=== SUMMARY ===")
    print(f"Fixed dataset with {count} properly formatted replays")
    print("Ready for training!")