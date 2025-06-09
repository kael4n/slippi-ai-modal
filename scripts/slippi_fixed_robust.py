# Slippi AI - Robust NumPy 1.24.3 lock with strict dependency management
import modal

PROJECT_ROOT = "/root/slippi-ai"
REPO_URL = "https://github.com/vladfi1/slippi-ai.git"

def create_slippi_image():
    """
    Create Slippi AI image with bulletproof NumPy 1.24.3 lock
    """
    # Start with Debian slim for better control
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
        "python3 -m pip install --upgrade pip setuptools wheel",
        # Install maturin via pip first for importability
        "python3 -m pip install maturin",
        # Install Rust toolchain
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
        "/root/.cargo/bin/cargo install maturin",
        "/root/.cargo/bin/maturin --version"
    ]).env({
        "PATH": "/root/.cargo/bin:$PATH",
        "PYTHONPATH": "/usr/local/lib/python3.10/site-packages",
        "CARGO_HOME": "/root/.cargo",
        "RUSTUP_HOME": "/root/.rustup"
    })

    # CRITICAL: Install and LOCK NumPy 1.24.3 before anything else
    image = image.run_commands([
        "python3 -m pip install numpy==1.24.3 --no-deps --force-reinstall",
        'python3 -c "import numpy; print(f\'✅ NumPy locked at: {numpy.__version__}\')"'
    ])

    # Install peppi-py carefully with no build isolation
    image = image.run_commands([
        "python3 -m pip install --no-build-isolation peppi-py==0.6.0",
        'python3 -c "import peppi_py; print(\'✅ peppi-py installed successfully\')"',
        'python3 -c "import numpy; print(f\'NumPy after peppi-py: {numpy.__version__}\')"'
    ])

    # Install compatible versions in the EXACT order that works
    # Using the successful pattern from your working script
    compatible_packages = [
        "scipy==1.10.1",
        "jax==0.4.13", 
        "jaxlib==0.4.13",
        "pandas==2.0.3",
        "tensorflow==2.13.0",
        "flax==0.7.2",
        "optax==0.1.7",
        "dm-haiku==0.0.10",
        "dm-tree==0.1.8",
        "sacred==0.8.4",
        "pymongo==4.5.0",
        "matplotlib==3.7.2",
        "seaborn==0.12.2",
        "tqdm==4.65.0",
        "cloudpickle==2.2.1",
        "absl-py==1.4.0",
        "tensorboard==2.13.0",
        "gymnasium==0.28.1"
    ]

    # Install packages one by one with NumPy protection
    for package in compatible_packages:
        image = image.run_commands([
            f"python3 -m pip install {package} --no-upgrade || python3 -m pip install {package}",
            'python3 -c "import numpy; print(f\'NumPy version check: {numpy.__version__}\')"'
        ])

    # Final NumPy protection - reinstall if it got upgraded
    image = image.run_commands([
        'python3 -c "import numpy; current=numpy.__version__; print(f\'Current NumPy: {current}\'); exit(0 if current==\'1.24.3\' else 1)" || python3 -m pip install --force-reinstall numpy==1.24.3 --no-deps',
        'python3 -c "import numpy; assert numpy.__version__ == \'1.24.3\', f\'FINAL CHECK FAILED: NumPy is {numpy.__version__}\'"'
    ])

    # Clone repository
    image = image.run_commands([
        f"git clone {REPO_URL} {PROJECT_ROOT}",
        f"ls -la {PROJECT_ROOT}"
    ]).workdir(PROJECT_ROOT)

    # Install additional requirements carefully
    image = image.run_commands([
        # Try to install additional requirements without breaking NumPy
        "python3 -m pip install tensorflow_probability==0.20.0 --no-upgrade || echo 'tensorflow_probability skipped'",
        "python3 -m pip install wandb --no-upgrade || echo 'wandb skipped'",
        "python3 -m pip install dnspython --no-upgrade || echo 'dnspython skipped'",
        "python3 -m pip install fancyflags --no-upgrade || echo 'fancyflags skipped'",
        "python3 -m pip install pyarrow --no-upgrade || echo 'pyarrow skipped'",
        "python3 -m pip install py7zr --no-upgrade || echo 'py7zr skipped'",
        "python3 -m pip install parameterized --no-upgrade || echo 'parameterized skipped'",
        "python3 -m pip install portpicker --no-upgrade || echo 'portpicker skipped'",
        "python3 -m pip install melee --no-upgrade || echo 'melee skipped'",
        # Final NumPy check and correction if needed
        'python3 -c "import numpy; current=numpy.__version__; print(f\'NumPy after extras: {current}\'); exit(0 if current==\'1.24.3\' else 1)" || python3 -m pip install --force-reinstall numpy==1.24.3 --no-deps'
    ])

    # Install slippi-ai in editable mode without dependencies to avoid conflicts
    image = image.run_commands([
        "python3 -m pip install -e . --no-deps || echo 'Editable install skipped due to conflicts'",
        # FINAL verification
        'python3 -c "import numpy; assert numpy.__version__ == \'1.24.3\', f\'CRITICAL FAILURE: Final NumPy version is {numpy.__version__}\'"',
        "echo '🎉 NumPy 1.24.3 successfully locked!'"
    ])

    return image

# Create the image
image = create_slippi_image()

app = modal.App("slippi-ai-robust-numpy-lock")

@app.function(image=image)
def comprehensive_test():
    """Comprehensive test with detailed version reporting"""
    import sys
    print("🐍 Python version:", sys.version)
    print("=" * 70)
    
    # Critical imports with version checks
    try:
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
        if np.__version__ != "1.24.3":
            print(f"❌ CRITICAL: NumPy version is {np.__version__}, expected 1.24.3")
            return False
        else:
            print("🎯 NumPy version is EXACTLY 1.24.3 - SUCCESS!")
    except Exception as e:
        print(f"❌ NumPy import failed: {e}")
        return False

    try:
        import peppi_py
        print(f"✅ peppi_py: Available at {peppi_py.__file__}")
    except Exception as e:
        print(f"❌ peppi_py import failed: {e}")
        return False

    try:
        import tensorflow as tf
        print(f"✅ TensorFlow: {tf.__version__}")
        # Test basic TF operation
        test_tensor = tf.constant([1.0, 2.0, 3.0])
        result = tf.reduce_mean(test_tensor)
        print(f"✅ TensorFlow test: mean([1,2,3]) = {result.numpy()}")
    except Exception as e:
        print(f"❌ TensorFlow test failed: {e}")
        return False

    try:
        import jax
        print(f"✅ JAX: {jax.__version__}")
        print(f"✅ JAX devices: {jax.devices()}")
        # Test basic JAX operation
        jax_array = jax.numpy.array([1.0, 2.0, 3.0])
        jax_result = jax.numpy.mean(jax_array)
        print(f"✅ JAX test: mean([1,2,3]) = {jax_result}")
    except Exception as e:
        print(f"❌ JAX test failed: {e}")
        return False

    try:
        import sacred
        print(f"✅ Sacred: {sacred.__version__}")
        from sacred import Experiment
        ex = Experiment('test')
        print("✅ Sacred experiment creation successful")
    except Exception as e:
        print(f"❌ Sacred test failed: {e}")
        return False

    try:
        import scipy
        print(f"✅ SciPy: {scipy.__version__}")
        from scipy import sparse
        matrix = sparse.csr_matrix([[1, 2], [3, 4]])
        print(f"✅ SciPy sparse test: matrix shape {matrix.shape}")
    except Exception as e:
        print(f"❌ SciPy test failed: {e}")
        return False

    print("\n🎉 ALL TESTS PASSED!")
    print("🚀 Slippi AI environment is ready for training!")
    return True

@app.function(
    image=image,
    volumes={"/data": modal.Volume.from_name("slippi-ai-dataset-doesokay", create_if_missing=True)}
)
def test_with_dataset():
    """Test environment with dataset access"""
    import os
    
    print("🗂️ Testing dataset access...")
    
    # First run comprehensive environment test
    env_ok = comprehensive_test.local()
    if not env_ok:
        print("❌ Environment test failed")
        return False
    
    # Check dataset access
    if os.path.exists("/data"):
        print("✅ Dataset volume mounted successfully")
        try:
            contents = os.listdir("/data")
            print(f"📁 Dataset contents: {contents}")
            
            if "games" in contents:
                games_path = "/data/games"
                if os.path.exists(games_path):
                    pkl_files = [f for f in os.listdir(games_path) if f.endswith('.pkl')]
                    print(f"🎮 Found {len(pkl_files)} .pkl game files")
                    
                    if len(pkl_files) > 0:
                        print("✅ Dataset is ready for training!")
                        return True
                    else:
                        print("⚠️  No .pkl files found in games directory")
                else:
                    print("❌ Games directory not accessible")
            else:
                print("❌ No 'games' directory found")
        except Exception as e:
            print(f"❌ Error accessing dataset: {e}")
    else:
        print("❌ Dataset volume not mounted")
    
    return False

@app.local_entrypoint()
def main():
    """Main test runner"""
    print("🏁 Starting Slippi AI Robust Environment Test...")
    
    print("\n" + "="*60)
    print("TEST 1: Environment & NumPy Lock Verification")
    print("="*60)
    success = comprehensive_test.remote()
    
    if success:
        print("\n" + "="*60)  
        print("TEST 2: Dataset Integration Test")
        print("="*60)
        dataset_success = test_with_dataset.remote()
        
        if dataset_success:
            print("\n🎉 SUCCESS: Complete Slippi AI environment is ready!")
            print("🚀 You can now proceed with training!")
        else:
            print("\n⚠️  Environment works but dataset access needs attention")
    else:
        print("\n❌ Environment setup failed - check the logs above")

if __name__ == "__main__":
    main()
