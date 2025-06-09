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

app = modal.App("slippi-dataset-organizer")
dataset_volume = modal.Volume.from_name(DATASET_VOLUME_NAME)
nfs = modal.NetworkFileSystem.from_name("slippi-ai-dataset-doesokay")

class PickleGameStub:
    """Stub class to handle peppi_py.game objects during unpickling"""
    def __init__(self, *args, **kwargs):
        self.data = {"args": args, "kwargs": kwargs}
        
    def __setstate__(self, state):
        self.data = state
        
    def __getstate__(self):
        return getattr(self, 'data', {})

# Create a custom unpickler that handles missing modules
class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Handle the missing peppi_py.game module
        if module == 'peppi_py.game' or module.startswith('peppi_py'):
            return PickleGameStub
        return super().find_class(module, name)

def safe_pickle_load(file_path):
    """Safely load a pickle file even if some modules are missing"""
    try:
        with open(file_path, 'rb') as f:
            unpickler = SafeUnpickler(f)
            return unpickler.load()
    except Exception as e:
        print(f"    └─ Safe unpickler also failed: {e}")
        return None

@app.function(
    image=image,
    volumes={"/dataset": dataset_volume},
    network_file_systems={"/nfs": nfs},
    timeout=3600,
)
def inspect_all_data():
    """Inspect all available data sources."""
    print("=== Inspecting All Data Sources ===")
    
    # 1. Check NFS contents
    print("\n📁 NFS Contents (/nfs):")
    nfs_path = Path("/nfs")
    if nfs_path.exists():
        item_count = 0
        for item in nfs_path.rglob("*"):
            if item_count > 20:  # Limit output
                print("  ... (more items)")
                break
            if item.is_file():
                size = item.stat().st_size
                print(f"  📄 {item.relative_to(nfs_path)} ({size:,} bytes)")
            elif item.is_dir():
                try:
                    file_count = len(list(item.iterdir()))
                    print(f"  📁 {item.relative_to(nfs_path)}/ ({file_count} items)")
                except:
                    print(f"  📁 {item.relative_to(nfs_path)}/")
            item_count += 1
    else:
        print("  ❌ NFS not mounted or empty")
    
    # 2. Check dataset volume contents
    print("\n💾 Dataset Volume Contents (/dataset):")
    dataset_path = Path("/dataset")
    if dataset_path.exists():
        item_count = 0
        for item in dataset_path.rglob("*"):
            if item_count > 20:  # Limit output
                print("  ... (more items)")
                break
            if item.is_file():
                size = item.stat().st_size
                print(f"  📄 {item.relative_to(dataset_path)} ({size:,} bytes)")
                
                # If it's a .pkl file, try to inspect it with both methods
                if item.suffix == '.pkl' and item_count < 5:  # Only try first few
                    print(f"    └─ Trying to load PKL...")
                    
                    # Try normal pickle first
                    try:
                        with open(item, 'rb') as f:
                            data = pickle.load(f)
                        print(f"    └─ Normal PKL type: {type(data)}")
                        if hasattr(data, '__len__'):
                            print(f"    └─ Normal PKL length: {len(data)}")
                    except Exception as e:
                        print(f"    └─ Normal pickle failed: {str(e)[:50]}...")
                        
                        # Try safe unpickler
                        data = safe_pickle_load(item)
                        if data is not None:
                            print(f"    └─ Safe PKL type: {type(data)}")
                            if hasattr(data, 'data'):
                                print(f"    └─ Safe PKL data keys: {list(data.data.keys()) if isinstance(data.data, dict) else 'not dict'}")
                        
            elif item.is_dir():
                try:
                    file_count = len(list(item.iterdir()))
                    print(f"  📁 {item.relative_to(dataset_path)}/ ({file_count} items)")
                except:
                    print(f"  📁 {item.relative_to(dataset_path)}/")
            item_count += 1
    else:
        print("  ❌ Dataset volume not mounted or empty")
    
    # 3. Check current meta.json
    print("\n📋 Current meta.json:")
    meta_file = Path("/dataset/meta.json")
    if meta_file.exists():
        try:
            with open(meta_file, 'r') as f:
                meta = json.load(f)
            print(f"  ✅ Found meta.json: {json.dumps(meta, indent=2)}")
        except Exception as e:
            print(f"  ❌ Error reading meta.json: {e}")
    else:
        print("  ❌ meta.json not found")

@app.function(
    image=image,
    volumes={"/dataset": dataset_volume},
    network_file_systems={"/nfs": nfs},
    timeout=3600,
)
def extract_replays_from_nfs():
    """Extract .slp files from the NFS zip file."""
    print("=== Extracting Replays from NFS ===")
    
    nfs_zip = Path("/nfs/replays.zip")
    nfs_replays_dir = Path("/nfs/replays")
    dataset_games_dir = Path("/dataset/games")
    
    # Create games directory
    dataset_games_dir.mkdir(exist_ok=True)
    
    extracted_count = 0
    
    # Try to extract from zip file first
    if nfs_zip.exists():
        print(f"📦 Extracting from {nfs_zip}")
        try:
            with zipfile.ZipFile(nfs_zip, 'r') as zip_ref:
                # List contents first
                file_list = zip_ref.namelist()
                slp_files = [f for f in file_list if f.endswith('.slp')]
                print(f"  Found {len(slp_files)} .slp files in zip")
                
                # Extract .slp files
                for slp_file in slp_files:
                    try:
                        # Extract to games directory
                        zip_ref.extract(slp_file, dataset_games_dir)
                        extracted_count += 1
                        if extracted_count <= 5:  # Show first 5
                            print(f"  ✅ Extracted {slp_file}")
                        elif extracted_count == 6:
                            print(f"  ... (continuing extraction)")
                    except Exception as e:
                        print(f"  ❌ Error extracting {slp_file}: {e}")
                        
        except Exception as e:
            print(f"❌ Error extracting zip: {e}")
    
    # Also check replays directory
    if nfs_replays_dir.exists():
        print(f"📁 Copying from {nfs_replays_dir}")
        slp_files = list(nfs_replays_dir.rglob("*.slp"))
        print(f"  Found {len(slp_files)} .slp files in directory")
        
        for slp_file in slp_files:
            try:
                # Copy to games directory, preserving structure
                rel_path = slp_file.relative_to(nfs_replays_dir)
                dest_path = dataset_games_dir / rel_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(slp_file, dest_path)
                extracted_count += 1
                if extracted_count <= 5:  # Show first 5
                    print(f"  ✅ Copied {rel_path}")
                elif extracted_count == 6:
                    print(f"  ... (continuing copy)")
            except Exception as e:
                print(f"  ❌ Error copying {slp_file}: {e}")
    
    print(f"📊 Total .slp files extracted/copied: {extracted_count}")
    return extracted_count

@app.function(
    image=image,
    volumes={"/dataset": dataset_volume},
    timeout=3600,
)
def convert_pkl_to_json():
    """Convert .pkl files to JSON format expected by slippi-ai."""
    print("=== Converting PKL files to JSON ===")
    
    os.chdir(PROJECT_ROOT)
    
    # Find all .pkl files
    pkl_files = list(Path("/dataset").rglob("*.pkl"))
    print(f"Found {len(pkl_files)} .pkl files")
    
    if not pkl_files:
        print("No .pkl files found to convert")
        return 0
    
    # Create Parsed directory
    parsed_dir = Path("/dataset/Parsed")
    parsed_dir.mkdir(exist_ok=True)
    
    converted_count = 0
    game_list = []
    
    for pkl_file in pkl_files:
        try:
            if converted_count <= 5:
                print(f"Converting {pkl_file.name}...")
            elif converted_count == 6:
                print("... (continuing conversion)")
            
            # Try to load pickle file with safe unpickler
            data = safe_pickle_load(pkl_file)
            
            if data is None:
                print(f"  ❌ Could not load {pkl_file.name}")
                continue
            
            # Create JSON structure expected by slippi-ai
            json_data = {
                'filename': pkl_file.stem + '.slp',  # Pretend it's from an .slp file
                'source_pkl': str(pkl_file.name),
                'meta': {},
                'data': None
            }
            
            # Try to extract useful information from the pickle data
            if hasattr(data, 'data') and isinstance(data.data, dict):
                # This is our stub object
                json_data['meta'] = {k: str(v)[:100] if isinstance(v, str) else str(v) 
                                   for k, v in data.data.items() if k != 'frames'}
                if 'frames' in data.data:
                    frames = data.data['frames']
                    json_data['frames'] = len(frames) if hasattr(frames, '__len__') else 'unknown'
            elif isinstance(data, dict):
                json_data['meta'] = {k: str(v)[:100] if isinstance(v, str) else str(v) 
                                   for k, v in data.items() if k != 'frames'}
                if 'frames' in data:
                    json_data['frames'] = len(data['frames']) if hasattr(data['frames'], '__len__') else 'unknown'
            elif hasattr(data, '__dict__'):
                json_data['meta'] = {k: str(v)[:100] if isinstance(v, str) else str(v) 
                                   for k, v in data.__dict__.items()}
            elif hasattr(data, '__len__'):
                json_data['frames'] = len(data)
                json_data['meta'] = {'type': str(type(data)), 'length': len(data)}
            else:
                json_data['meta'] = {'type': str(type(data)), 'content': str(data)[:100]}
            
            # Save as JSON
            json_file = parsed_dir / f"{pkl_file.stem}.json"
            with open(json_file, 'w') as f:
                json.dump(json_data, f, indent=2, default=str)
            
            game_list.append(pkl_file.stem + '.slp')
            converted_count += 1
            
            if converted_count <= 5:
                print(f"  ✅ Converted {pkl_file.name}")
                
        except Exception as e:
            print(f"  ❌ Error converting {pkl_file.name}: {e}")
    
    # Update meta.json
    meta_data = {
        'games': game_list,
        'total_games': len(game_list),
        'parsed_dir': '/dataset/Parsed',
        'source': 'converted_from_pkl'
    }
    
    with open('/dataset/meta.json', 'w') as f:
        json.dump(meta_data, f, indent=2)
    
    print(f"✅ Converted {converted_count} PKL files to JSON")
    print(f"✅ Updated meta.json with {len(game_list)} games")
    return converted_count

@app.function(
    image=image,
    volumes={"/dataset": dataset_volume},
    timeout=3600,
)
def parse_slp_files():
    """Parse any .slp files found in the dataset."""
    print("=== Parsing SLP Files ===")
    
    os.chdir(PROJECT_ROOT)
    
    try:
        import peppi
        print("✅ Peppi available for parsing")
    except ImportError:
        print("❌ Peppi not available - cannot parse .slp files")
        return 0
    
    # Find all .slp files
    slp_files = list(Path("/dataset/games").rglob("*.slp"))
    print(f"Found {len(slp_files)} .slp files")
    
    if not slp_files:
        print("No .slp files found to parse")
        return 0
    
    # Create Parsed directory
    parsed_dir = Path("/dataset/Parsed")
    parsed_dir.mkdir(exist_ok=True)
    
    parsed_count = 0
    game_list = []
    
    for slp_file in slp_files[:50]:  # Limit to first 50 for testing
        try:
            if parsed_count <= 5:
                print(f"Parsing {slp_file.name}...")
            elif parsed_count == 6:
                print("... (continuing parsing)")
                
            # Parse with peppi
            game = peppi.read_slippi(str(slp_file))
            
            # Create JSON structure
            json_data = {
                'filename': slp_file.name,
                'source_slp': str(slp_file.relative_to(Path('/dataset'))),
                'frames': len(game.frames) if hasattr(game, 'frames') else 0,
                'meta': {}
            }
            
            # Extract metadata
            if hasattr(game, 'metadata'):
                try:
                    if hasattr(game.metadata, '_asdict'):
                        json_data['meta'] = game.metadata._asdict()
                    else:
                        json_data['meta'] = str(game.metadata)
                except:
                    json_data['meta'] = {'raw_metadata': str(game.metadata)}
            
            # Save as JSON
            json_file = parsed_dir / f"{slp_file.stem}.json"
            with open(json_file, 'w') as f:
                json.dump(json_data, f, indent=2, default=str)
            
            game_list.append(slp_file.name)
            parsed_count += 1
            
        except Exception as e:
            if parsed_count <= 5:
                print(f"  ❌ Error parsing {slp_file.name}: {e}")
    
    # Update meta.json
    meta_data = {
        'games': game_list,
        'total_games': len(game_list),
        'parsed_dir': '/dataset/Parsed',
        'source': 'parsed_from_slp'
    }
    
    with open('/dataset/meta.json', 'w') as f:
        json.dump(meta_data, f, indent=2)
    
    print(f"✅ Parsed {parsed_count} SLP files to JSON")
    print(f"✅ Updated meta.json with {len(game_list)} games")
    return parsed_count

@app.local_entrypoint()
def main():
    print("=== Slippi Dataset Organizer ===")
    print("This will help organize your dataset from multiple sources")
    
    # First, inspect everything
    print("\n1. Inspecting all data sources...")
    inspect_all_data.remote()
    
    # Since interactive input doesn't work well in Modal, let's do everything
    print("\n2. Running all organization tasks...")
    print("Doing everything...")
    extract_count = extract_replays_from_nfs.remote()
    pkl_count = convert_pkl_to_json.remote()
    slp_count = parse_slp_files.remote()
    
    print(f"\n=== SUMMARY ===")
    print(f"  - Extracted: {extract_count} files from NFS")
    print(f"  - Converted PKL: {pkl_count} files") 
    print(f"  - Parsed SLP: {slp_count} files")
    print(f"\nTotal processed files: {extract_count + pkl_count + slp_count}")
    print("\nAfter organizing, you can test your dataset with the training script!")

# Alternative entrypoints for specific tasks
@app.local_entrypoint()
def inspect_only():
    """Just inspect the data sources"""
    print("=== Inspection Only ===")
    inspect_all_data.remote()

@app.local_entrypoint() 
def convert_pkl_only():
    """Convert PKL files to JSON only"""
    print("=== Converting PKL Files Only ===")
    count = convert_pkl_to_json.remote()
    print(f"Converted {count} PKL files")

@app.local_entrypoint()
def extract_slp_only():
    """Extract SLP files from NFS only""" 
    print("=== Extracting SLP Files Only ===")
    count = extract_replays_from_nfs.remote()
    print(f"Extracted {count} SLP files")

@app.local_entrypoint()
def parse_slp_only():
    """Parse SLP files to JSON only"""
    print("=== Parsing SLP Files Only ===") 
    count = parse_slp_files.remote()
    print(f"Parsed {count} SLP files")