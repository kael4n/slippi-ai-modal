import modal
import os
import json
import shutil
from pathlib import Path
import requests
import zipfile
from typing import List

# Use the same volumes and setup
DATASET_VOLUME_NAME = "slippi-ai-dataset-doesokay"
PROJECT_ROOT = "/root/slippi-ai"
REPO_URL = "https://github.com/vladfi1/slippi-ai.git"
PEPPI_PY_COMMIT_URL = "git+https://github.com/hohav/peppi-py.git@8c02a4659c3302321dfbfcf2093c62f634e335f7"

# Same image as your training script
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "build-essential", "pkg-config", "libssl-dev", "curl", "wget", "unzip")
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

app = modal.App("slippi-dataset-uploader")
dataset_volume = modal.Volume.from_name(DATASET_VOLUME_NAME)

@app.function(
    image=image,
    volumes={"/dataset": dataset_volume},
    timeout=7200,  # 2 hours
)
def download_sample_replays():
    """Download sample Slippi replays for testing."""
    print("=== Downloading Sample Replays ===")
    
    games_dir = Path("/dataset/games")
    games_dir.mkdir(exist_ok=True)
    
    # Sample URLs for Slippi replays (you'll need to replace these with actual URLs)
    # These are just examples - you need real .slp file URLs
    sample_urls = [
        # Add actual .slp file URLs here
        # "https://example.com/replay1.slp",
        # "https://example.com/replay2.slp",
    ]
    
    if not sample_urls:
        print("⚠️  No sample URLs provided. Please add actual .slp file URLs to the script.")
        print("You can:")
        print("1. Find .slp files online")
        print("2. Upload your own replays")
        print("3. Use the slippi.gg API to download replays")
        return
    
    downloaded = 0
    for i, url in enumerate(sample_urls):
        try:
            print(f"Downloading replay {i+1}/{len(sample_urls)}...")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            filename = f"replay_{i+1:03d}.slp"
            filepath = games_dir / filename
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            print(f"✅ Downloaded {filename} ({len(response.content)} bytes)")
            downloaded += 1
            
        except Exception as e:
            print(f"❌ Failed to download {url}: {e}")
    
    print(f"Downloaded {downloaded} replays")
    return downloaded

@app.function(
    image=image,
    volumes={"/dataset": dataset_volume},
    timeout=3600,
)
def create_toy_dataset():
    """Create a minimal toy dataset for testing."""
    print("=== Creating Toy Dataset ===")
    
    try:
        import sys
        sys.path.append(PROJECT_ROOT)
        from slippi_ai import data
        
        # Check if toy_data_source is available
        if hasattr(data, 'toy_data_source'):
            print("✅ Found toy_data_source function")
            
            # Create toy dataset
            print("Creating toy dataset...")
            toy_source = data.toy_data_source()
            
            # Save to dataset directory
            games_dir = Path("/dataset/games")
            parsed_dir = Path("/dataset/Parsed")
            games_dir.mkdir(exist_ok=True)
            parsed_dir.mkdir(exist_ok=True)
            
            # Generate some sample data
            print("Generating sample data...")
            sample_count = 0
            
            try:
                # Try to get a few samples from toy source
                for i, batch in enumerate(toy_source):
                    if i >= 10:  # Just create 10 samples
                        break
                    
                    # Create a fake .slp filename
                    fake_filename = f"toy_game_{i:03d}.slp"
                    
                    # Create minimal parsed data
                    parsed_data = {
                        'meta': {
                            'game_id': f"toy_{i}",
                            'filename': fake_filename,
                            'player_count': 2,
                        },
                        'frames': len(batch) if hasattr(batch, '__len__') else 100,
                        'toy_data': True
                    }
                    
                    # Save parsed data
                    parsed_file = parsed_dir / f"toy_game_{i:03d}.json"
                    with open(parsed_file, 'w') as f:
                        json.dump(parsed_data, f, default=str)
                    
                    sample_count += 1
                    print(f"Created toy sample {i+1}")
                
            except Exception as e:
                print(f"Error generating toy data: {e}")
            
            # Update meta.json
            meta_data = {
                'games': [f"toy_game_{i:03d}.slp" for i in range(sample_count)],
                'total_games': sample_count,
                'parsed_dir': '/dataset/Parsed',
                'toy_dataset': True
            }
            
            with open('/dataset/meta.json', 'w') as f:
                json.dump(meta_data, f, indent=2)
            
            print(f"✅ Created toy dataset with {sample_count} samples")
            return sample_count
            
        else:
            print("❌ toy_data_source not found in slippi_ai.data")
            return 0
            
    except Exception as e:
        print(f"❌ Error creating toy dataset: {e}")
        import traceback
        traceback.print_exc()
        return 0

@app.function(
    image=image,
    volumes={"/dataset": dataset_volume},
    timeout=3600,
)
def parse_existing_replays():
    """Parse any existing .slp files in the dataset."""
    print("=== Parsing Existing Replays ===")
    
    os.chdir(PROJECT_ROOT)
    
    try:
        import peppi
        
        games_dir = Path("/dataset/games")
        parsed_dir = Path("/dataset/Parsed")
        parsed_dir.mkdir(exist_ok=True)
        
        # Find all .slp files
        slp_files = list(games_dir.glob("**/*.slp"))
        print(f"Found {len(slp_files)} .slp files")
        
        if not slp_files:
            print("No .slp files found to parse")
            return 0
        
        parsed_count = 0
        game_list = []
        
        for slp_file in slp_files:
            try:
                print(f"Parsing {slp_file.name}...")
                
                # Parse with peppi
                game = peppi.read_slippi(str(slp_file))
                
                # Extract relevant data
                parsed_data = {
                    'filename': slp_file.name,
                    'frames': len(game.frames) if hasattr(game, 'frames') else 0,
                    'meta': {},
                }
                
                # Try to extract metadata
                if hasattr(game, 'metadata'):
                    try:
                        if hasattr(game.metadata, '_asdict'):
                            parsed_data['meta'] = game.metadata._asdict()
                        else:
                            parsed_data['meta'] = str(game.metadata)
                    except:
                        parsed_data['meta'] = {'raw_metadata': str(game.metadata)}
                
                # Save parsed data
                output_file = parsed_dir / f"{slp_file.stem}.json"
                with open(output_file, 'w') as f:
                    json.dump(parsed_data, f, default=str, indent=2)
                
                game_list.append(slp_file.name)
                parsed_count += 1
                print(f"✅ Parsed {slp_file.name}")
                
            except Exception as e:
                print(f"❌ Error parsing {slp_file.name}: {e}")
        
        # Update meta.json
        meta_data = {
            'games': game_list,
            'total_games': len(game_list),
            'parsed_dir': '/dataset/Parsed'
        }
        
        with open('/dataset/meta.json', 'w') as f:
            json.dump(meta_data, f, indent=2)
        
        print(f"✅ Successfully parsed {parsed_count} replays")
        return parsed_count
        
    except Exception as e:
        print(f"❌ Error in parsing: {e}")
        import traceback
        traceback.print_exc()
        return 0

@app.function(
    image=image,
    volumes={"/dataset": dataset_volume},
    timeout=3600,
)
def verify_dataset():
    """Verify the dataset is ready for training."""
    print("=== Dataset Verification ===")
    
    try:
        import sys
        sys.path.append(PROJECT_ROOT)
        from slippi_ai import data
        
        # Check files
        meta_file = Path("/dataset/meta.json")
        parsed_dir = Path("/dataset/Parsed")
        
        if not meta_file.exists():
            print("❌ meta.json missing")
            return False
        
        with open(meta_file, 'r') as f:
            meta = json.load(f)
        
        print(f"📊 Dataset stats:")
        print(f"  - Total games: {meta.get('total_games', 0)}")
        print(f"  - Games in list: {len(meta.get('games', []))}")
        
        parsed_files = list(parsed_dir.glob("*.json"))
        print(f"  - Parsed files: {len(parsed_files)}")
        
        if len(parsed_files) == 0:
            print("❌ No parsed files found")
            return False
        
        # Try to create a data source
        print("🔧 Testing data loading...")
        try:
            # Try the replays_from_meta function
            if hasattr(data, 'replays_from_meta'):
                replays = data.replays_from_meta(str(meta_file))
                print(f"✅ Successfully loaded {len(replays)} replays with replays_from_meta")
                return True
            else:
                print("❌ replays_from_meta function not found")
                
        except Exception as e:
            print(f"❌ Error loading with replays_from_meta: {e}")
        
        # Try other loading methods
        try:
            if hasattr(data, 'make_source'):
                source = data.make_source('/dataset')
                print(f"✅ Successfully created data source with make_source")
                return True
        except Exception as e:
            print(f"❌ Error with make_source: {e}")
        
        return False
        
    except Exception as e:
        print(f"❌ Verification error: {e}")
        import traceback
        traceback.print_exc()
        return False

@app.local_entrypoint()
def main():
    print("=== Slippi Dataset Setup ===")
    print("Choose an option:")
    print("1. Create toy dataset (for testing)")
    print("2. Parse existing .slp files (if you have them)")
    print("3. Download sample replays (requires URLs)")
    print("4. Verify current dataset")
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == "1":
        print("Creating toy dataset...")
        count = create_toy_dataset.remote()
        if count > 0:
            print(f"✅ Created toy dataset with {count} samples")
            print("Verifying dataset...")
            verify_dataset.remote()
        
    elif choice == "2":
        print("Parsing existing replays...")
        count = parse_existing_replays.remote()
        if count > 0:
            print("Verifying dataset...")
            verify_dataset.remote()
        
    elif choice == "3":
        print("Note: You need to add actual .slp URLs to the script first")
        download_sample_replays.remote()
        
    elif choice == "4":
        print("Verifying dataset...")
        verify_dataset.remote()
        
    else:
        print("Invalid choice")