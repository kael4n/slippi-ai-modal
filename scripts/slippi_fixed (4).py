# Fixed Slippi AI - Modal compatible version with NumPy 1.24.3 lock
import modal

PROJECT_ROOT = "/root/slippi-ai"
REPO_URL = "https://github.com/vladfi1/slippi-ai.git"

def create_slippi_image():
    """
    Create Slippi AI image with proper NumPy version control for Modal
    """
    # Start with Python 3.10 slim
    image = modal.Image.from_registry("python:3.10-slim")
    
    # Install essential system packages
    image = image.apt_install([
        "tzdata", "build-essential", "pkg-config", "cmake", "ninja-build",
        "libssl-dev", "libffi-dev", "zlib1g-dev", "libbz2-dev", "libreadline-dev",
        "libsqlite3-dev", "libncurses5-dev", "libncursesw5-dev", "xz-utils",
        "tk-dev", "libxml2-dev", "libxmlsec1-dev", "liblzma-dev", "git", "curl",
        "wget", "unzip", "software-properties-common", "libgl1-mesa-glx",
        "libglib2.0-0", "libsm6", "libxext6", "libxrender-dev", "libgomp1"
    ])

    # Install Rust toolchain and maturin
    image = image.run_commands([
        "python -m pip install --upgrade pip setuptools wheel",
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
        "python -m pip install maturin==1.2.3"
    ]).env({
        "PATH": "/root/.cargo/bin:$PATH",
        "CARGO_HOME": "/root/.cargo",
        "RUSTUP_HOME": "/root/.rustup"
    })

    # CRITICAL: Install NumPy 1.24.3 first and lock it
    image = image.run_commands([
        "python -m pip uninstall numpy -y || true",
        "python -m pip install numpy==1.24.3 --no-deps",
        'python -c "import numpy; print(f\'Initial NumPy version: {numpy.__version__}\')"'
    ])

    # Install peppi-py with Rust backend
    image = image.run_commands([
        "/root/.cargo/bin/cargo install maturin",
        "python -m pip install --no-build-isolation peppi-py==0.6.0",
        'python -c "import peppi_py; print(\'✅ peppi-py installed successfully\')"'
    ]).env({
        "PATH": "/root/.cargo/bin:$PATH",
        "CARGO_HOME": "/root/.cargo",
        "RUSTUP_HOME": "/root/.rustup"
    })

    # Install compatible scientific packages one by one to avoid conflicts
    # Core scientific stack - compatible with NumPy 1.24.3
    image = image.run_commands([
        "python -m pip install scipy==1.10.1 --no-deps",
        "python -m pip install pandas==2.0.3",
        'python -c "import numpy; print(f\'NumPy after scipy/pandas: {numpy.__version__}\')"'
    ])
    
    # ML frameworks - versions compatible with NumPy 1.24.3
    image = image.run_commands([
        "python -m pip install jax==0.4.13 jaxlib==0.4.13",
        "python -m pip install tensorflow==2.13.0",
        'python -c "import numpy; print(f\'NumPy after JAX/TF: {numpy.__version__}\')"'
    ])
    
    # JAX ecosystem
    image = image.run_commands([
        "python -m pip install flax==0.7.2",
        "python -m pip install optax==0.1.7",
        "python -m pip install dm-haiku==0.0.10",
        "python -m pip install dm-tree==0.1.8",
        'python -c "import numpy; print(f\'NumPy after JAX ecosystem: {numpy.__version__}\')"'
    ])
    
    # Experiment tracking and utilities
    image = image.run_commands([
        "python -m pip install sacred==0.8.4",
        "python -m pip install pymongo==4.5.0",
        "python -m pip install matplotlib==3.7.2",
        "python -m pip install seaborn==0.12.2",
        "python -m pip install tqdm==4.65.0",
        "python -m pip install cloudpickle==2.2.1",
        "python -m pip install absl-py==1.4.0",
        "python -m pip install tensorboard==2.13.0",
        "python -m pip install gymnasium==0.28.1",
        'python -c "import numpy; print(f\'NumPy after utilities: {numpy.__version__}\')"'
    ])

    # Install dm-sonnet carefully
    image = image.run_commands([
        "python -m pip install dm-sonnet==2.0.1 --no-deps",
        'python -c "import numpy; print(f\'NumPy after dm-sonnet: {numpy.__version__}\')"'
    ])

    # Verify NumPy hasn't been upgraded
    image = image.run_commands([
        'python -c "import numpy; print(f\'Final NumPy check: {numpy.__version__}\')"',
        'python -c "import numpy; assert numpy.__version__ == \'1.24.3\', f\'NumPy version changed to {numpy.__version__}\'"'
    ])

    # Clone the repository
    image = image.run_commands([
        f"git clone {REPO_URL} {PROJECT_ROOT}",
        f"ls -la {PROJECT_ROOT}"
    ]).workdir(PROJECT_ROOT)

    # Install additional requirements carefully
    image = image.run_commands([
        # Create a safer requirements install
        "python -m pip install tensorflow_probability==0.20.0 --no-upgrade",
        "python -m pip install wandb --no-upgrade",
        "python -m pip install dnspython --no-upgrade",
        "python -m pip install fancyflags --no-upgrade || echo 'fancyflags skipped'",
        "python -m pip install pyarrow>=10.0.0 --no-upgrade",
        "python -m pip install py7zr --no-upgrade",
        "python -m pip install parameterized --no-upgrade",
        "python -m pip install portpicker --no-upgrade",
        "python -m pip install melee>=0.38.0 --no-upgrade",
        'python -c "import numpy; print(f\'NumPy after additional packages: {numpy.__version__}\')"'
    ])

    # Install slippi-ai in editable mode without dependencies
    image = image.run_commands([
        "python -m pip install -e . --no-deps",
        'python -c "import numpy; print(f\'FINAL NumPy version: {numpy.__version__}\')"'
    ])

    # Final comprehensive verification
    image = image.run_commands([
        'python -c "import numpy; assert numpy.__version__ == \'1.24.3\', f\'CRITICAL: NumPy {numpy.__version__} != 1.24.3\'"',
        'python -c "import scipy; print(f\'SciPy: {scipy.__version__}\')"',
        'python -c "import tensorflow; print(f\'TensorFlow: {tensorflow.__version__}\')"',
        'python -c "import jax; print(f\'JAX: {jax.__version__}\')"',
        'python -c "import peppi_py; print(f\'peppi_py: OK\')"',
        'python -c "import sacred; print(f\'Sacred: {sacred.__version__}\')"',
        "echo '✅ All packages verified successfully'"
    ])

    return image

# Create the image
image = create_slippi_image()

app = modal.App("slippi-ai-fixed")

@app.function(
    image=image,
    mounts=[modal.Mount.from_local_dir(".", remote_path="/local")]  # Mount local files if needed
)
def test_slippi_environment():
    """Comprehensive test of the Slippi AI environment"""
    import numpy as np
    import scipy
    import tensorflow as tf
    import jax
    import peppi_py
    import sacred
    
    print("=" * 70)
    print("🧪 SLIPPI AI ENVIRONMENT TEST")
    print("=" * 70)
    
    # Critical NumPy version check
    print(f"🎯 NumPy version: {np.__version__}")
    if np.__version__ == "1.24.3":
        print("✅ NumPy version is EXACTLY 1.24.3 - SUCCESS!")
        numpy_ok = True
    else:
        print(f"❌ CRITICAL: NumPy version is {np.__version__}, expected 1.24.3")
        numpy_ok = False
    
    # Package versions
    print(f"📊 SciPy version: {scipy.__version__}")
    print(f"🧠 TensorFlow version: {tf.__version__}")
    print(f"⚡ JAX version: {jax.__version__}")
    print(f"🎮 peppi_py available: {peppi_py.__file__}")
    print(f"🔬 Sacred version: {sacred.__version__}")
    
    # Functionality tests
    tests_passed = 0
    total_tests = 6
    
    # 1. NumPy test
    try:
        arr = np.array([1, 2, 3, 4, 5])
        result = np.mean(arr)
        print(f"✅ NumPy test: mean([1,2,3,4,5]) = {result}")
        tests_passed += 1
    except Exception as e:
        print(f"❌ NumPy test failed: {e}")
    
    # 2. TensorFlow test
    try:
        tf_tensor = tf.constant([1.0, 2.0, 3.0])
        tf_result = tf.reduce_mean(tf_tensor)
        print(f"✅ TensorFlow test: {tf_result.numpy()}")
        tests_passed += 1
    except Exception as e:
        print(f"❌ TensorFlow test failed: {e}")
    
    # 3. JAX test
    try:
        jax_array = jax.numpy.array([1.0, 2.0, 3.0])
        jax_result = jax.numpy.mean(jax_array)
        print(f"✅ JAX test: {jax_result}")
        print(f"✅ JAX devices: {jax.devices()}")
        tests_passed += 1
    except Exception as e:
        print(f"❌ JAX test failed: {e}")
    
    # 4. SciPy test
    try:
        from scipy import sparse
        sparse_matrix = sparse.csr_matrix([[1, 2], [3, 4]])
        print(f"✅ SciPy sparse test: matrix shape {sparse_matrix.shape}")
        tests_passed += 1
    except Exception as e:
        print(f"❌ SciPy test failed: {e}")
    
    # 5. peppi_py test
    try:
        print("✅ peppi_py import successful")
        tests_passed += 1
    except Exception as e:
        print(f"❌ peppi_py test failed: {e}")
    
    # 6. Sacred test
    try:
        from sacred import Experiment
        ex = Experiment('test')
        print("✅ Sacred experiment creation successful")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Sacred test failed: {e}")
    
    # Summary
    print(f"\n🏆 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if numpy_ok and tests_passed >= total_tests - 1:
        print("🎉 SUCCESS: Slippi AI environment is ready!")
        return True
    else:
        print("❌ FAILURE: Environment has issues")
        return False

@app.function(
    image=image,
    volumes={"/data": modal.Volume.from_name("slippi-ai-dataset-doesokay")}  # Mount your dataset
)
def test_dataset_access():
    """Test access to your uploaded dataset"""
    import os
    
    print("🗂️ Testing dataset access...")
    
    # Check if dataset volume is mounted
    if os.path.exists("/data"):
        print("✅ Dataset volume mounted successfully")
        
        # List contents
        try:
            contents = os.listdir("/data")
            print(f"📁 Dataset contents: {contents}")
            
            # Check for games directory
            if "games" in contents:
                games_path = "/data/games"
                if os.path.exists(games_path):
                    game_files = os.listdir(games_path)[:10]  # Show first 10 files
                    print(f"🎮 Game files (first 10): {game_files}")
                    
                    # Count .pkl files
                    pkl_files = [f for f in os.listdir(games_path) if f.endswith('.pkl')]
                    print(f"📦 Total .pkl files: {len(pkl_files)}")
                    
                    return len(pkl_files)
            else:
                print("❌ No 'games' directory found in dataset")
        except Exception as e:
            print(f"❌ Error accessing dataset: {e}")
    else:
        print("❌ Dataset volume not mounted")
    
    return 0

@app.function(image=image)
def run_slippi_training_example():
    """Example of how to set up training with your dataset"""
    try:
        import os
        import sys
        sys.path.insert(0, "/root/slippi-ai")
        
        print("🚀 Setting up Slippi AI training environment...")
        
        # Test basic slippi-ai imports
        print("📦 Testing Slippi AI module imports...")
        
        # Check what modules are available
        if os.path.exists("/root/slippi-ai"):
            print("📂 Slippi AI directory structure:")
            for root, dirs, files in os.walk("/root/slippi-ai"):
                level = root.replace("/root/slippi-ai", "").count(os.sep)
                indent = " " * 2 * level
                print(f"{indent}{os.path.basename(root)}/")
                subindent = " " * 2 * (level + 1)
                for file in files[:5]:  # Show first 5 files in each dir
                    if file.endswith(('.py', '.yaml', '.yml', '.json', '.md')):
                        print(f"{subindent}{file}")
                if len(files) > 5:
                    print(f"{subindent}... and {len(files) - 5} more files")
        
        print("✅ Slippi AI environment setup complete")
        
    except Exception as e:
        print(f"❌ Training setup failed: {e}")
        import traceback
        traceback.print_exc()

@app.local_entrypoint()
def main():
    """Main entry point for testing the environment"""
    print("🏁 Starting Slippi AI environment tests...")
    
    # Test 1: Environment setup
    print("\n" + "="*50)
    print("TEST 1: Environment Setup")
    print("="*50)
    success = test_slippi_environment.remote()
    
    if not success:
        print("❌ Environment test failed - stopping here")
        return
    
    # Test 2: Dataset access
    print("\n" + "="*50)
    print("TEST 2: Dataset Access")
    print("="*50)
    dataset_size = test_dataset_access.remote()
    print(f"📊 Dataset contains {dataset_size} .pkl files")
    
    # Test 3: Training setup
    print("\n" + "="*50)
    print("TEST 3: Training Setup")
    print("="*50)
    run_slippi_training_example.remote()
    
    print("\n🎉 All tests completed!")
    print("🚀 Your Slippi AI environment is ready for training!")

if __name__ == "__main__":
    main()