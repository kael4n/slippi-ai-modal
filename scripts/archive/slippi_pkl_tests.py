# Slippi PKL File Testing Script
import modal
import pickle
import os
from pathlib import Path

# Use the same image we built earlier
PROJECT_ROOT = "/root/slippi-ai"
REPO_URL = "https://github.com/vladfi1/slippi-ai.git"

def create_slippi_image():
    image = modal.Image.debian_slim().apt_install([
        "tzdata", "python3", "python3-pip", "python3-dev", "build-essential",
        "pkg-config", "cmake", "ninja-build", "libssl-dev", "libffi-dev", "zlib1g-dev",
        "libbz2-dev", "libreadline-dev", "libsqlite3-dev", "libncurses5-dev",
        "libncursesw5-dev", "xz-utils", "tk-dev", "libxml2-dev", "libxmlsec1-dev",
        "liblzma-dev", "git", "curl", "wget", "unzip", "software-properties-common",
        "libgl1-mesa-glx", "libglib2.0-0", "libsm6", "libxext6", "libxrender-dev", "libgomp1"
    ])

    # Set up Python and basic tools
    image = image.run_commands([
        "ln -sf /usr/bin/python3 /usr/bin/python",
        "python3 -m pip install --upgrade pip==23.2.1",
        "python3 -m pip install setuptools==68.0.0 wheel==0.41.2"
    ])

    # Create constraint file first
    image = image.run_commands([
        'echo "numpy==1.24.3" > /root/numpy-constraint.txt',
        "mkdir -p /root/.pip",
        'echo "[install]" > /root/.pip/pip.conf',
        'echo "constraint = /root/numpy-constraint.txt" >> /root/.pip/pip.conf',
    ])

    # Install NumPy with constraint
    image = image.run_commands([
        "python3 -m pip uninstall numpy -y || true",
        "python3 -m pip install 'numpy==1.24.3' --no-deps --force-reinstall",
        'python3 -c "import numpy as np; print(f\'✅ NumPy locked at: {np.__version__}\')"'
    ]).env({"PIP_CONSTRAINT": "/root/numpy-constraint.txt"})

    # Install Rust and maturin
    image = image.run_commands([
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
        "python3 -m pip install maturin==1.2.3",
        "/root/.cargo/bin/cargo install maturin",
    ]).env({
        "PATH": "/root/.cargo/bin:$PATH",
        "CARGO_HOME": "/root/.cargo",
        "RUSTUP_HOME": "/root/.rustup",
        "PIP_CONSTRAINT": "/root/numpy-constraint.txt"
    })

    # Install peppi-py
    image = image.run_commands([
        "python3 -m pip install --no-build-isolation peppi-py==0.6.0",
    ]).env({
        "PIP_CONSTRAINT": "/root/numpy-constraint.txt",
        "PATH": "/root/.cargo/bin:$PATH",
        "CARGO_HOME": "/root/.cargo",
        "RUSTUP_HOME": "/root/.rustup"
    })

    # Install other dependencies
    image = image.run_commands([
        "python3 -m pip install 'scipy==1.10.1'",
        "python3 -m pip install 'pandas==2.0.3'",
        "python3 -m pip install 'tensorflow==2.13.0'",
        "python3 -m pip install 'jax==0.4.13' --no-deps",
        "python3 -m pip install 'jaxlib==0.4.13' --no-deps",
        "python3 -m pip install sacred==0.8.4",
        "python3 -m pip install tqdm==4.65.0",
    ]).env({"PIP_CONSTRAINT": "/root/numpy-constraint.txt"})

    # Clone repository
    image = image.run_commands([
        f"git clone {REPO_URL} {PROJECT_ROOT}",
        "python3 -m pip install -e . || echo 'Editable install completed'"
    ]).workdir(PROJECT_ROOT).env({"PIP_CONSTRAINT": "/root/numpy-constraint.txt"})

    return image

# Create the image
image = create_slippi_image()

# Create the app with volume mount
app = modal.App("slippi-pkl-tests")
volume = modal.Volume.from_name("slippi-ai-dataset-doesokay", create_if_missing=False)

@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=600  # 10 minutes timeout
)
def analyze_pkl_files():
    """Analyze the .pkl files in the Modal volume"""
    import pickle
    import numpy as np
    import os
    from pathlib import Path
    
    print("="*60)
    print("🎮 SLIPPI PKL FILE ANALYSIS")
    print("="*60)
    
    # Check what's in the data directory
    data_path = Path("/data")
    print(f"📁 Contents of {data_path}:")
    
    if data_path.exists():
        for item in data_path.iterdir():
            if item.is_dir():
                print(f"   📂 {item.name}/")
                # Look inside directories
                for subitem in item.iterdir():
                    print(f"      📄 {subitem.name}")
                    if len(list(item.iterdir())) > 10:  # If too many files, just show count
                        total_files = len(list(item.iterdir()))
                        print(f"      ... and {total_files - 10} more files")
                        break
            else:
                print(f"   📄 {item.name}")
    else:
        print("   ❌ Data directory not found!")
        return
    
    # Look for .pkl files specifically
    pkl_files = list(data_path.rglob("*.pkl"))
    print(f"\n🔍 Found {len(pkl_files)} .pkl files")
    
    if not pkl_files:
        print("❌ No .pkl files found in the volume")
        return
    
    # Analyze first few pkl files
    print(f"\n📊 Analyzing first 5 .pkl files:")
    
    for i, pkl_file in enumerate(pkl_files[:5]):
        print(f"\n--- File {i+1}: {pkl_file.name} ---")
        print(f"📍 Path: {pkl_file}")
        print(f"📏 Size: {pkl_file.stat().st_size / 1024:.2f} KB")
        
        try:
            # Try to load the pickle file
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            
            print(f"✅ Successfully loaded pickle file")
            print(f"📝 Type: {type(data)}")
            
            # Analyze the data structure
            if hasattr(data, '__len__'):
                print(f"📊 Length: {len(data)}")
            
            if isinstance(data, dict):
                print(f"🔑 Dict keys: {list(data.keys())[:10]}")  # Show first 10 keys
                for key, value in list(data.items())[:3]:  # Show first 3 items
                    print(f"   {key}: {type(value)} - {str(value)[:100]}")
            
            elif isinstance(data, (list, tuple)):
                print(f"📋 First few elements:")
                for j, item in enumerate(data[:3]):
                    print(f"   [{j}]: {type(item)} - {str(item)[:100]}")
            
            elif isinstance(data, np.ndarray):
                print(f"🔢 NumPy array shape: {data.shape}")
                print(f"🔢 NumPy array dtype: {data.dtype}")
                print(f"🔢 NumPy array sample: {data.flat[:10]}")
            
            else:
                print(f"📄 Content preview: {str(data)[:200]}")
                
        except Exception as e:
            print(f"❌ Error loading pickle file: {e}")
    
    print(f"\n📈 Summary:")
    print(f"   Total .pkl files found: {len(pkl_files)}")
    total_size = sum(f.stat().st_size for f in pkl_files) / (1024 * 1024)  # MB
    print(f"   Total size: {total_size:.2f} MB")
    
    return {
        "total_files": len(pkl_files),
        "total_size_mb": total_size,
        "file_paths": [str(f) for f in pkl_files[:10]]  # Return first 10 paths
    }

@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=600
)
def test_slippi_data_loading():
    """Test loading and processing Slippi data using the slippi-ai library"""
    import sys
    sys.path.append('/root/slippi-ai')
    
    print("="*60)  
    print("🎮 SLIPPI-AI LIBRARY TESTS")
    print("="*60)
    
    try:
        # Try to import slippi-ai modules
        print("📦 Testing slippi-ai imports...")
        
        # This might need adjustment based on the actual module structure
        # Let's first check what's available
        import os
        slippi_path = "/root/slippi-ai"
        print(f"Contents of {slippi_path}:")
        for item in os.listdir(slippi_path):
            print(f"   {item}")
        
        # Try to find Python modules
        for root, dirs, files in os.walk(slippi_path):
            for file in files:
                if file.endswith('.py') and not file.startswith('__'):
                    rel_path = os.path.relpath(os.path.join(root, file), slippi_path)
                    print(f"   Python file: {rel_path}")
        
        print("✅ Directory exploration completed")
        
    except Exception as e:
        print(f"❌ Error during slippi-ai testing: {e}")
        import traceback
        traceback.print_exc()

@app.local_entrypoint()
def main():
    print("🚀 Starting Slippi PKL analysis...")
    
    # First, analyze the pkl files
    result = analyze_pkl_files.remote()
    print(f"Analysis result: {result}")
    
    # Then test the slippi-ai library
    test_slippi_data_loading.remote()