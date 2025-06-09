import modal
import os
import json
from pathlib import Path

# Use the same volumes as your training script
DATASET_VOLUME_NAME = "slippi-ai-dataset-doesokay"
PROJECT_ROOT = "/root/slippi-ai"
REPO_URL = "https://github.com/vladfi1/slippi-ai.git"
PEPPI_PY_COMMIT_URL = "git+https://github.com/hohav/peppi-py.git@8c02a4659c3302321dfbfcf2093c62f634e335f7"

# Same image as your training script
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

app = modal.App("slippi-data-debugger-enhanced")
dataset_volume = modal.Volume.from_name(DATASET_VOLUME_NAME)

@app.function(
    image=image,
    volumes={"/dataset": dataset_volume},
    timeout=3600,
)
def enhanced_debug():
    """Enhanced debug to understand why parsing failed and attempt to fix it."""
    os.chdir(PROJECT_ROOT)
    
    print("=== Enhanced Dataset Debug ===")
    
    # 1. Check meta.json in detail
    meta_file = Path("/dataset/meta.json")
    if meta_file.exists():
        print("📄 Meta.json analysis:")
        with open(meta_file, 'r') as f:
            meta_content = json.load(f)
        
        print(f"  - Total games: {meta_content.get('total_games', 'N/A')}")
        print(f"  - Games list length: {len(meta_content.get('games', []))}")
        print(f"  - Parsed dir: {meta_content.get('parsed_dir', 'N/A')}")
        
        # Show first few games
        games = meta_content.get('games', [])
        if games:
            print(f"  - First 3 games: {games[:3]}")
    
    # 2. Check games directory
    games_dir = Path("/dataset/games")
    if games_dir.exists():
        print(f"\n🎮 Games directory analysis:")
        slp_files = list(games_dir.glob("*.slp"))
        print(f"  - .slp files found: {len(slp_files)}")
        
        if slp_files:
            print(f"  - First few .slp files:")
            for slp_file in slp_files[:5]:
                size = slp_file.stat().st_size
                print(f"    * {slp_file.name} ({size} bytes)")
        else:
            # Check subdirectories
            subdirs = [d for d in games_dir.iterdir() if d.is_dir()]
            print(f"  - Subdirectories: {[d.name for d in subdirs]}")
            
            # Look for .slp files in subdirectories
            total_slp = 0
            for subdir in subdirs[:3]:  # Check first 3 subdirs
                subdir_slp = list(subdir.glob("**/*.slp"))
                total_slp += len(subdir_slp)
                print(f"    * {subdir.name}: {len(subdir_slp)} .slp files")
            print(f"  - Total .slp files in subdirs (first 3): {total_slp}")
    
    # 3. Check what slippi-ai modules are available
    print(f"\n🔍 Available slippi-ai modules:")
    try:
        import sys
        sys.path.append(PROJECT_ROOT)
        import slippi_ai
        print(f"  - slippi_ai location: {slippi_ai.__file__}")
        print(f"  - slippi_ai modules: {dir(slippi_ai)}")
        
        # Check if data module exists
        try:
            from slippi_ai import data
            print(f"  - data module: {dir(data)}")
        except ImportError as e:
            print(f"  - data module import error: {e}")
        
        # Check other modules
        for module_name in ['dataset', 'parse', 'replay']:
            try:
                module = getattr(slippi_ai, module_name, None)
                if module:
                    print(f"  - {module_name} module: {dir(module)}")
            except Exception as e:
                print(f"  - {module_name} module error: {e}")
                
    except Exception as e:
        print(f"❌ Error importing slippi_ai: {e}")
    
    # 4. Try to find parsing functions
    print(f"\n🔧 Looking for parsing functions:")
    try:
        # Look in different possible locations
        possible_modules = ['slippi_ai.data', 'slippi_ai.dataset', 'slippi_ai.parse']
        
        for module_name in possible_modules:
            try:
                parts = module_name.split('.')
                module = __import__(module_name, fromlist=[parts[-1]])
                functions = [name for name in dir(module) if callable(getattr(module, name)) and not name.startswith('_')]
                print(f"  - {module_name}: {functions}")
            except ImportError:
                print(f"  - {module_name}: not found")
    except Exception as e:
        print(f"❌ Error exploring modules: {e}")
    
    # 5. Try to manually parse a single file
    print(f"\n🎯 Attempting manual parsing:")
    try:
        # Find a .slp file to test with
        games_dir = Path("/dataset/games")
        slp_files = list(games_dir.glob("**/*.slp"))
        
        if slp_files:
            test_file = slp_files[0]
            print(f"  - Testing with: {test_file}")
            
            # Try using peppi directly
            try:
                import peppi
                print(f"  - peppi version: {peppi.__version__ if hasattr(peppi, '__version__') else 'unknown'}")
                
                # Try to parse the file
                game = peppi.read_slippi(str(test_file))
                print(f"  - Successfully parsed with peppi!")
                print(f"  - Game type: {type(game)}")
                print(f"  - Game attributes: {[attr for attr in dir(game) if not attr.startswith('_')]}")
                
            except Exception as e:
                print(f"  - peppi parsing error: {e}")
                
        else:
            print("  - No .slp files found to test with")
            
    except Exception as e:
        print(f"❌ Error in manual parsing: {e}")
    
    # 6. Try to run the actual parsing step
    print(f"\n⚙️ Attempting to run parsing step:")
    try:
        # Look for scripts that might do the parsing
        script_candidates = [
            "scripts/parse_dataset.py",
            "parse_dataset.py", 
            "scripts/parse.py",
            "parse.py"
        ]
        
        for script in script_candidates:
            script_path = Path(PROJECT_ROOT) / script
            if script_path.exists():
                print(f"  - Found parsing script: {script}")
                # Don't actually run it yet, just report
            else:
                print(f"  - Script not found: {script}")
                
        # Look for parsing functionality in main modules
        print("  - Checking for parse functions in slippi_ai...")
        
    except Exception as e:
        print(f"❌ Error checking parsing scripts: {e}")
    
    print(f"\n=== Debug Complete ===")

@app.function(
    image=image,
    volumes={"/dataset": dataset_volume},
    timeout=3600,
)
def attempt_parsing():
    """Attempt to actually parse the dataset."""
    os.chdir(PROJECT_ROOT)
    
    print("=== Attempting Dataset Parsing ===")
    
    try:
        import sys
        sys.path.append(PROJECT_ROOT)
        
        # Try to find and run the parsing
        games_dir = Path("/dataset/games")
        parsed_dir = Path("/dataset/Parsed")
        
        # Ensure parsed directory exists
        parsed_dir.mkdir(exist_ok=True)
        
        # Find .slp files
        slp_files = list(games_dir.glob("**/*.slp"))
        print(f"Found {len(slp_files)} .slp files to parse")
        
        if slp_files:
            # Try parsing with peppi
            import peppi
            parsed_count = 0
            
            for i, slp_file in enumerate(slp_files[:10]):  # Parse first 10 files
                try:
                    print(f"Parsing {slp_file.name}...")
                    game = peppi.read_slippi(str(slp_file))
                    
                    # Convert to JSON and save
                    output_file = parsed_dir / f"{slp_file.stem}.json"
                    
                    # Convert peppi game object to dict
                    game_dict = {
                        'meta': game.metadata._asdict() if hasattr(game.metadata, '_asdict') else str(game.metadata),
                        'frames': len(game.frames) if hasattr(game, 'frames') else 0,
                        'file': str(slp_file.name)
                    }
                    
                    with open(output_file, 'w') as f:
                        json.dump(game_dict, f, default=str)
                    
                    parsed_count += 1
                    
                except Exception as e:
                    print(f"  - Error parsing {slp_file.name}: {e}")
            
            print(f"Successfully parsed {parsed_count} files")
        
    except Exception as e:
        print(f"❌ Parsing attempt failed: {e}")
        import traceback
        traceback.print_exc()

@app.local_entrypoint()
def main():
    print("Starting enhanced dataset debug...")
    enhanced_debug.remote()
    
    # Ask user if they want to attempt parsing
    response = input("\nWould you like to attempt parsing the dataset? (y/n): ")
    if response.lower().startswith('y'):
        print("Attempting to parse dataset...")
        attempt_parsing.remote()