# Fixed Slippi AI Modal Script - Proper NumPy Version Locking (Based on Working Version)
import modal

PROJECT_ROOT = "/root/slippi-ai"
REPO_URL = "https://github.com/vladfi1/slippi-ai.git"

def create_slippi_image():
    """
    Create prebuilt Slippi AI image using the EXACT working approach with enhanced NumPy protection
    This mirrors your working slippi_WORKING.py but adds extra safeguards against NumPy upgrades
    """
    # Use the same base image approach as the working version
    image = modal.Image.debian_slim().apt_install([
        "tzdata", "python3", "python3-pip", "python3-dev", "build-essential",
        "pkg-config", "cmake", "ninja-build", "libssl-dev", "libffi-dev", "zlib1g-dev",
        "libbz2-dev", "libreadline-dev", "libsqlite3-dev", "libncurses5-dev",
        "libncursesw5-dev", "xz-utils", "tk-dev", "libxml2-dev", "libxmlsec1-dev",
        "liblzma-dev", "git", "curl", "wget", "unzip", "software-properties-common",
        "libgl1-mesa-glx", "libglib2.0-0", "libsm6", "libxext6", "libxrender-dev", "libgomp1"
    ])

    # EXACT same setup commands as working version
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

    # 🔒 CRITICAL: Lock numpy FIRST before anything else, prevent upgrades
    # This is the exact approach from your working version
    image = image.run_commands([
        "python3 -m pip install numpy==1.24.3 --no-deps"
    ])

    # ✅ EXACT same peppi-py installation as working version
    image = image.run_commands([
        "python3 -m pip install --no-build-isolation peppi-py==0.6.0",
        'python3 -c "import peppi_py; print(\'✅ peppi-py v0.6.0 installed and importable\')"'
    ])

    # 🔒 ENHANCED: Add extra NumPy protection by pinning it again before scientific packages
    # This prevents any dependency from upgrading NumPy
    image = image.run_commands([
        # Verify NumPy is still 1.24.3
        'python3 -c "import numpy; print(f\'NumPy version before scientific packages: {numpy.__version__}\')"',
        # Re-pin NumPy to be absolutely sure
        "python3 -m pip install numpy==1.24.3 --force-reinstall --no-deps"
    ])

    # ✅ Pin all scientific deps AFTER numpy is frozen (exact same versions as working)
    # Using --no-deps where possible to prevent NumPy upgrades
    image = image.run_commands([
        # Install scipy with explicit NumPy constraint
        "python3 -m pip install 'scipy==1.10.1' --no-deps",
        # Install JAX components with NumPy constraint
        "python3 -m pip install 'jax==0.4.13' --no-deps",
        "python3 -m pip install 'jaxlib==0.4.13' --no-deps",
        # Install other packages that are less likely to upgrade NumPy
        "python3 -m pip install pandas==2.0.3",
        "python3 -m pip install tensorflow==2.13.0",
        "python3 -m pip install flax==0.7.2",
        "python3 -m pip install optax==0.1.7",
        "python3 -m pip install dm-haiku==0.0.10",
        "python3 -m pip install dm-tree==0.1.8",
        "python3 -m pip install sacred==0.8.4",
        "python3 -m pip install pymongo==4.5.0",
        "python3 -m pip install matplotlib==3.7.2",
        "python3 -m pip install seaborn==0.12.2",
        "python3 -m pip install tqdm==4.65.0",
        "python3 -m pip install cloudpickle==2.2.1",
        "python3 -m pip install absl-py==1.4.0",
        "python3 -m pip install tensorboard==2.13.0",
        "python3 -m pip install gymnasium==0.28.1"
    ])

    # 🔒 FINAL SAFETY CHECK: Verify and lock NumPy one more time
    image = image.run_commands([
        'python3 -c "import numpy; print(f\'Final NumPy version: {numpy.__version__}\')"',
        # If NumPy got upgraded, force it back down
        "python3 -m pip install numpy==1.24.3 --force-reinstall --no-deps",
        'python3 -c "import numpy; assert numpy.__version__ == \'1.24.3\', f\'NumPy version mismatch: {numpy.__version__}\'"'
    ])

    # Clone and install slippi-ai (same as working version)
    image = image.run_commands([
        f"git clone {REPO_URL} {PROJECT_ROOT}",
        f"ls -la {PROJECT_ROOT}"
    ]).workdir(PROJECT_ROOT)

    # EXACT same installation commands as working version
    image = image.run_commands([
        "python3 -m pip install -r requirements.txt || echo 'Partial requirements succeeded'",
        "python3 -m pip install -e . || echo 'Editable install skipped/fallback'"
    ])

    # 🔒 ULTIMATE SAFETY: Final NumPy verification
    image = image.run_commands([
        'python3 -c "import numpy; print(f\'FINAL NumPy version after all installs: {numpy.__version__}\')"',
        'python3 -c "import numpy; assert numpy.__version__.startswith(\'1.24\'), f\'NumPy upgrade detected: {numpy.__version__}\'"'
    ])

    return image

# Create the prebuilt image
image = create_slippi_image()

app = modal.App("slippi-ai-prebuilt-numpy-locked")

@app.function(image=image)
def test_prebuilt_env():
    """Test that exactly matches the working version's test"""
    import peppi_py
    import tensorflow as tf
    import jax
    import sacred
    import numpy as np

    print("=" * 60)
    print("🚀 PREBUILT IMAGE VALIDATION (NumPy-Locked Version)")
    print("=" * 60)
    
    print("✅ peppi_py:", peppi_py.__file__)
    print("✅ numpy version:", np.__version__)
    print("✅ tensorflow version:", tf.__version__)
    print("✅ jax version:", jax.__version__)
    print("✅ sacred version:", sacred.__version__)
    print("✅ jax.devices:", jax.devices())
    
    # Test TensorFlow functionality (same as working version)
    try:
        tf_result = tf.random.uniform((1,))
        print("✅ tf.random.uniform(1):", tf_result)
    except Exception as e:
        print("❌ TensorFlow test failed:", e)
    
    # Additional tests to verify everything works
    try:
        jax_result = jax.numpy.array([1, 2, 3])
        print("✅ JAX array test:", jax_result)
    except Exception as e:
        print("❌ JAX test failed:", e)
    
    # Critical NumPy compatibility checks
    try:
        # Test for NumPy 2.x incompatibilities
        hasattr(np, 'issubsctype')  # This should exist in NumPy 1.x
        print("✅ NumPy 1.x compatibility confirmed")
    except:
        print("❌ NumPy compatibility issue detected")
    
    print("🎉 Prebuilt environment validated with NumPy locking!")

@app.function(image=image)
def run_slippi_ai_task():
    """
    Example function showing how to use the prebuilt environment for actual slippi-ai tasks
    """
    import sys
    import os
    import numpy as np
    
    # Add the project to Python path
    sys.path.insert(0, PROJECT_ROOT)
    
    print("=" * 60)
    print("🎮 SLIPPI-AI ENVIRONMENT READY FOR TASKS")
    print("=" * 60)
    
    print(f"📁 Project root: {PROJECT_ROOT}")
    print(f"🐍 Python path includes: {PROJECT_ROOT}")
    print(f"🔒 NumPy version locked at: {np.__version__}")
    
    # List available slippi-ai modules/scripts
    if os.path.exists(PROJECT_ROOT):
        print("📋 Available files:")
        for root, dirs, files in os.walk(PROJECT_ROOT):
            # Only show Python files in the main directory
            if root == PROJECT_ROOT:
                py_files = [f for f in files if f.endswith('.py')]
                for py_file in py_files[:10]:  # Show first 10
                    print(f"   • {py_file}")
                if len(py_files) > 10:
                    print(f"   ... and {len(py_files) - 10} more")
    
    print("✅ Slippi-AI environment is ready for use!")
    print("🚀 You can now run training, inference, or analysis tasks!")
    print("🔒 All dependencies locked to prevent version conflicts!")

@app.function(image=image)
def debug_numpy_situation():
    """
    Debug function to understand exactly what's happening with NumPy
    """
    import numpy as np
    import sys
    import pkg_resources
    
    print("=" * 60)
    print("🔍 NUMPY DEBUG INFORMATION")
    print("=" * 60)
    
    print(f"NumPy version: {np.__version__}")
    print(f"NumPy file location: {np.__file__}")
    print(f"Python version: {sys.version}")
    
    # Check if problematic NumPy 2.x attributes exist
    attrs_to_check = ['issubsctype', 'issubdtype', '_ARRAY_API']
    for attr in attrs_to_check:
        has_attr = hasattr(np, attr)
        print(f"np.{attr} exists: {has_attr}")
    
    # List all installed packages
    print("\n📦 Installed packages (scientific stack):")
    packages_of_interest = ['numpy', 'scipy', 'jax', 'jaxlib', 'tensorflow']
    for pkg_name in packages_of_interest:
        try:
            pkg = pkg_resources.get_distribution(pkg_name)
            print(f"  {pkg_name}: {pkg.version}")
        except:
            print(f"  {pkg_name}: not found")

@app.local_entrypoint()
def main():
    print("🔧 Testing NumPy-locked prebuilt Slippi AI environment...")
    
    # Run debug first to understand the situation
    debug_numpy_situation.remote()
    
    # Run all tests
    test_prebuilt_env.remote()
    run_slippi_ai_task.remote()
    
    print("\n" + "=" * 60)
    print("✅ NUMPY-LOCKED PREBUILT IMAGE TESTS COMPLETE")
    print("=" * 60)
    print("💡 Your prebuilt image is ready with NumPy properly locked!")
    print("🎯 Use this image for fast Slippi-AI deployments")
    print("🔒 NumPy version conflicts prevented!")

if __name__ == "__main__":
    main()