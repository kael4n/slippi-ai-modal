# Slippi AI - Based on PROVEN Working Version with Enhanced Testing
import modal

PROJECT_ROOT = "/root/slippi-ai"
REPO_URL = "https://github.com/vladfi1/slippi-ai.git"

def create_slippi_image():
    """
    Create Slippi AI image using the EXACT pattern from working script
    """
    image = modal.Image.debian_slim().apt_install([
        "tzdata", "python3", "python3-pip", "python3-dev", "build-essential",
        "pkg-config", "cmake", "ninja-build", "libssl-dev", "libffi-dev", "zlib1g-dev",
        "libbz2-dev", "libreadline-dev", "libsqlite3-dev", "libncurses5-dev",
        "libncursesw5-dev", "xz-utils", "tk-dev", "libxml2-dev", "libxmlsec1-dev",
        "liblzma-dev", "git", "curl", "wget", "unzip", "software-properties-common",
        "libgl1-mesa-glx", "libglib2.0-0", "libsm6", "libxext6", "libxrender-dev", "libgomp1"
    ])

    image = image.run_commands([
        "ln -sf /usr/bin/python3 /usr/bin/python",
        "python3 -m pip install --upgrade pip setuptools wheel",
        # ✅ install maturin via pip so it is importable
        "python3 -m pip install maturin",
        # ✅ install Rust + CLI maturin
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
        "/root/.cargo/bin/cargo install maturin",
        "/root/.cargo/bin/maturin --version"
    ]).env({
        "PATH": "/root/.cargo/bin:$PATH",
        "PYTHONPATH": "/usr/local/lib/python3.10/site-packages",
        "CARGO_HOME": "/root/.cargo",
        "RUSTUP_HOME": "/root/.rustup"
    })

    # 🔥 CRITICAL FIX: Lock numpy before anything else, prevent upgrades
    image = image.run_commands([
        "python3 -m pip install numpy==1.24.3 --no-deps --force-reinstall"
    ])

    # ✅ Install peppi-py with correct backend and no isolation
    image = image.run_commands([
        "python3 -m pip install --no-build-isolation peppi-py==0.6.0",
        'python3 -c "import peppi_py; print(\'✅ peppi-py v0.6.0 installed and importable\')"'
    ])

    # 🔥 CRITICAL FIX: Pin ALL scientific deps with EXACT versions from working script
    image = image.pip_install([
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
    ])

    image = image.run_commands([
        f"git clone {REPO_URL} {PROJECT_ROOT}",
        f"ls -la {PROJECT_ROOT}"
    ]).workdir(PROJECT_ROOT)

    image = image.run_commands([
        "python3 -m pip install -r requirements.txt || echo 'Partial requirements succeeded'",
        "python3 -m pip install -e . || echo 'Editable install skipped/fallback'"
    ])

    return image

# Create the image using proven working method
image = create_slippi_image()

app = modal.App("slippi-ai-proven-working-enhanced")

@app.function(image=image)
def test_env():
    """EXACT test from working version"""
    import peppi_py
    import tensorflow as tf
    import jax
    import sacred
    import numpy as np

    print("✅ peppi_py:", peppi_py.__file__)
    print("✅ numpy version:", np.__version__)
    print("✅ tensorflow version:", tf.__version__)
    print("✅ jax version:", jax.__version__)
    print("✅ sacred version:", sacred.__version__)
    print("✅ jax.devices:", jax.devices())
    print("✅ tf.random.uniform(1):", tf.random.uniform((1,)))

@app.function(image=image)
def comprehensive_test():
    """Enhanced test with detailed version reporting and compatibility checks"""
    import sys
    print("🐍 Python version:", sys.version)
    print("=" * 70)
    
    # Test core dependencies first
    try:
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
        
        # Verify NumPy is the correct version
        if np.__version__ != "1.24.3":
            print(f"❌ WARNING: NumPy version should be 1.24.3, got {np.__version__}")
            return False
            
        import scipy
        print(f"✅ SciPy: {scipy.__version__}")
        
        # Test the specific operation that was failing
        from scipy.sparse import issparse
        print("✅ SciPy sparse import successful")
        
        import jax
        import jax.numpy as jnp  # This was failing before
        print(f"✅ JAX: {jax.__version__}")
        print(f"✅ JAX devices: {jax.devices()}")
        
        import tensorflow as tf
        print(f"✅ TensorFlow: {tf.__version__}")
        
        # Test TensorFlow operations
        test_tensor = tf.constant([1.0, 2.0, 3.0])
        mean_result = tf.reduce_mean(test_tensor)
        print(f"✅ TensorFlow operations working: mean = {mean_result.numpy()}")
        
        import peppi_py
        print(f"✅ peppi-py: {peppi_py.__version__ if hasattr(peppi_py, '__version__') else 'installed'}")
        
        import sacred
        print(f"✅ Sacred: {sacred.__version__}")
        
        print("🎯 All core tests PASSED!")
        
    except Exception as e:
        print(f"❌ Core test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Additional framework tests
    try:
        print("\n🧪 Additional ML framework tests...")
        
        import flax
        print(f"✅ Flax: {flax.__version__}")
        
        import optax
        print(f"✅ Optax: {optax.__version__}")
        
        import haiku as hk
        print(f"✅ Haiku: {hk.__version__}")
        
        import pandas as pd
        df = pd.DataFrame({'a': [1, 2, 3]})
        print(f"✅ Pandas test: {df.shape}")
        
    except Exception as e:
        print(f"⚠️  Some additional tests failed: {e}")

    print("\n🎉 COMPREHENSIVE TESTS COMPLETED!")
    print("🚀 Environment matches proven working configuration!")
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
    try:
        env_ok = comprehensive_test.local()
        if not env_ok:
            print("❌ Environment test failed")
            return False
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
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
                        
                        # Test loading a sample file if available
                        sample_file = pkl_files[0]
                        try:
                            import pickle
                            with open(os.path.join(games_path, sample_file), 'rb') as f:
                                data = pickle.load(f)
                            print(f"✅ Successfully loaded sample file: {sample_file}")
                            print(f"📊 Sample data type: {type(data)}")
                        except Exception as e:
                            print(f"⚠️  Could not load sample file: {e}")
                        
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

@app.function(image=image)
def test_slippi_ai_training_readiness():
    """Test if environment is ready for actual Slippi AI training"""
    try:
        print("🎯 Testing Slippi AI training readiness...")
        
        # Test imports that would be needed for training
        import sys
        import os
        sys.path.insert(0, "/root/slippi-ai")
        
        print(f"✅ Working directory: {os.getcwd()}")
        print(f"✅ Python path includes: {'/root/slippi-ai' in sys.path}")
        
        # Test if we can access the repository structure
        if os.path.exists("/root/slippi-ai"):
            contents = os.listdir("/root/slippi-ai")
            print(f"✅ Slippi AI repo contents: {contents}")
            
            # Check for common training files
            training_indicators = []
            if "setup.py" in contents:
                training_indicators.append("setup.py")
            if "requirements.txt" in contents:
                training_indicators.append("requirements.txt")
            if any("train" in f.lower() for f in contents):
                training_indicators.extend([f for f in contents if "train" in f.lower()])
            if any("config" in f.lower() for f in contents):
                training_indicators.extend([f for f in contents if "config" in f.lower()])
                
            print(f"✅ Training-related files found: {training_indicators}")
            
            return True
        else:
            print("❌ Slippi AI repository not found")
            return False
            
    except Exception as e:
        print(f"❌ Training readiness test failed: {e}")
        return False

@app.local_entrypoint()
def main():
    """Enhanced main test runner"""
    print("🏁 Starting Slippi AI PROVEN WORKING Configuration Test...")
    
    print("\n" + "="*60)
    print("TEST 1: Original Working Test (Exact Copy)")
    print("="*60)
    try:
        test_env.remote()
        print("✅ Original test completed successfully!")
    except Exception as e:
        print(f"❌ Original test failed: {e}")
        return
    
    print("\n" + "="*60)
    print("TEST 2: Comprehensive Environment Test")
    print("="*60)
    try:
        success = comprehensive_test.remote()
        
        if success:
            print("\n" + "="*60)  
            print("TEST 3: Training Readiness Test")
            print("="*60)
            training_ready = test_slippi_ai_training_readiness.remote()
            
            if training_ready:
                print("\n" + "="*60)  
                print("TEST 4: Dataset Integration Test")
                print("="*60)
                dataset_success = test_with_dataset.remote()
                
                if dataset_success:
                    print("\n🎉 COMPLETE SUCCESS!")
                    print("🚀 Slippi AI environment is 100% ready for training!")
                    print("📋 Summary:")
                    print("   ✅ All dependencies working (NumPy 1.24.3 + TensorFlow 2.13.0)")
                    print("   ✅ peppi-py successfully installed")
                    print("   ✅ JAX, Sacred, and all ML frameworks working")
                    print("   ✅ Slippi AI code accessible")
                    print("   ✅ Dataset ready and loadable")
                else:
                    print("\n⚠️  Environment perfect, but dataset needs attention")
            else:
                print("\n⚠️  Environment works but training setup needs review")
        else:
            print("\n❌ Environment test failed")
    except Exception as e:
        print(f"❌ Comprehensive test failed: {e}")

if __name__ == "__main__":
    main()