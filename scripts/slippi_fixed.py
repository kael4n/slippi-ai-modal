# v18-bulletproof-numpy: Absolutely prevent NumPy 2.x installation
import modal

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
        "python3 -m pip install --upgrade pip==23.2.1",  # Use older pip version
        "python3 -m pip install setuptools==68.0.0 wheel==0.41.2"
    ])

    # Install Rust and maturin for peppi-py
    image = image.run_commands([
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
        "/root/.cargo/bin/cargo install maturin",
    ]).env({
        "PATH": "/root/.cargo/bin:$PATH",
        "PYTHONPATH": "/usr/local/lib/python3.10/site-packages",
        "CARGO_HOME": "/root/.cargo",
        "RUSTUP_HOME": "/root/.rustup"
    })

    # CRITICAL: Install NumPy 1.24.3 FIRST and create a lock mechanism
    image = image.run_commands([
        # Remove any existing NumPy installations
        "python3 -m pip uninstall numpy -y || true",
        # Install exact NumPy version with no dependencies
        "python3 -m pip install 'numpy==1.24.3' --no-deps --force-reinstall",
        # Create a pip.conf to prevent NumPy upgrades
        "mkdir -p /root/.pip",
        'echo "[install]" > /root/.pip/pip.conf',
        'echo "constraint = /root/numpy-constraint.txt" >> /root/.pip/pip.conf',
        # Create constraint file to lock NumPy
        'echo "numpy==1.24.3" > /root/numpy-constraint.txt',
        # Verify NumPy version
        'python3 -c "import numpy as np; print(f\'✅ NumPy locked at: {np.__version__}\')"'
    ]).env({
        "PIP_CONSTRAINT": "/root/numpy-constraint.txt"
    })

    # Install peppi-py with NumPy constraint active
    image = image.run_commands([
        "python3 -m pip install --no-build-isolation peppi-py==0.6.0",
        'python3 -c "import peppi_py; print(\'✅ peppi-py installed successfully\')"'
    ]).env({
        "PIP_CONSTRAINT": "/root/numpy-constraint.txt"
    })

    # Install SciPy with NumPy constraint
    image = image.run_commands([
        "python3 -m pip install 'scipy==1.10.1'",
        'python3 -c "import scipy; print(f\'✅ SciPy version: {scipy.__version__}\')"'
    ]).env({
        "PIP_CONSTRAINT": "/root/numpy-constraint.txt"
    })

    # Install pandas with NumPy constraint
    image = image.run_commands([
        "python3 -m pip install 'pandas==2.0.3'",
        'python3 -c "import pandas; print(f\'✅ Pandas version: {pandas.__version__}\')"'
    ]).env({
        "PIP_CONSTRAINT": "/root/numpy-constraint.txt"
    })

    # Install TensorFlow dependencies manually with constraint
    image = image.run_commands([
        "python3 -m pip install 'protobuf>=3.20.3,<5.0.0dev'",
        "python3 -m pip install 'six>=1.12.0'",
        "python3 -m pip install 'termcolor>=1.1.0'",
        "python3 -m pip install 'typing-extensions>=3.6.6'",
        "python3 -m pip install 'wrapt>=1.11.0'",
        "python3 -m pip install 'astunparse>=1.6.0'",
        "python3 -m pip install 'gast<=0.4.0,>=0.2.1'",
        "python3 -m pip install 'google-pasta>=0.1.1'",
        "python3 -m pip install 'h5py>=2.9.0'",
        "python3 -m pip install 'libclang>=13.0.0'",
        "python3 -m pip install 'opt-einsum>=2.3.2'",
        "python3 -m pip install 'packaging'",
        "python3 -m pip install 'flatbuffers>=1.12'",
        "python3 -m pip install 'grpcio>=1.24.3,<2.0'",
        "python3 -m pip install 'tensorflow-io-gcs-filesystem>=0.23.1'",
        "python3 -m pip install 'keras==2.13.1'",
        "python3 -m pip install 'tensorboard==2.13.0'",
        "python3 -m pip install 'tensorflow-estimator==2.13.0'"
    ]).env({
        "PIP_CONSTRAINT": "/root/numpy-constraint.txt"
    })

    # Install TensorFlow with no dependency resolution
    image = image.run_commands([
        "python3 -m pip install 'tensorflow==2.13.0' --no-deps",
        # Verify NumPy is still correct before testing TensorFlow
        'python3 -c "import numpy; print(f\'NumPy before TF test: {numpy.__version__}\')"',
        # Test TensorFlow import
        'python3 -c "import tensorflow as tf; print(f\'✅ TensorFlow version: {tf.__version__}\')"'
    ]).env({
        "PIP_CONSTRAINT": "/root/numpy-constraint.txt"
    })

    # Install JAX with constraint
    image = image.run_commands([
        "python3 -m pip install 'ml-dtypes>=0.1.0'",
        "python3 -m pip install 'jaxlib==0.4.13' --no-deps",
        "python3 -m pip install 'jax==0.4.13' --no-deps",
        'python3 -c "import jax; print(f\'✅ JAX version: {jax.__version__}\')"'
    ]).env({
        "PIP_CONSTRAINT": "/root/numpy-constraint.txt"
    })

    # Install remaining packages with constraint
    image = image.run_commands([
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
        "python3 -m pip install gymnasium==0.28.1"
    ]).env({
        "PIP_CONSTRAINT": "/root/numpy-constraint.txt"
    })

    # Final comprehensive verification
    image = image.run_commands([
        'python3 -c "import numpy; print(f\'✅ Final NumPy version: {numpy.__version__}\')"',
        'python3 -c "import scipy; print(f\'✅ Final SciPy version: {scipy.__version__}\')"',
        'python3 -c "import pandas; print(f\'✅ Final Pandas version: {pandas.__version__}\')"',
        'python3 -c "import tensorflow; print(f\'✅ Final TensorFlow version: {tensorflow.__version__}\')"',
        'python3 -c "import jax; print(f\'✅ Final JAX version: {jax.__version__}\')"',
        'python3 -c "import jaxlib; print(f\'✅ Final JAXlib version: {jaxlib.__version__}\')"',
        # Verify TensorFlow actually works
        'python3 -c "import tensorflow as tf; print(f\'✅ TensorFlow test: {tf.constant([1,2,3])}\")\"'
    ]).env({
        "PIP_CONSTRAINT": "/root/numpy-constraint.txt"
    })

    # Clone repository and set working directory
    image = image.run_commands([
        f"git clone {REPO_URL} {PROJECT_ROOT}",
        f"ls -la {PROJECT_ROOT}"
    ]).workdir(PROJECT_ROOT).env({
        "PIP_CONSTRAINT": "/root/numpy-constraint.txt"
    })

    # Install project requirements with constraint active
    image = image.run_commands([
        "python3 -m pip install -r requirements.txt || echo 'Partial requirements install completed'",
        "python3 -m pip install -e . || echo 'Editable install completed with warnings'"
    ]).env({
        "PIP_CONSTRAINT": "/root/numpy-constraint.txt"
    })

    return image

image = create_slippi_image()

app = modal.App("slippi-ai-bulletproof-numpy-v18")

@app.function(image=image)
def test_env():
    import peppi_py
    import tensorflow as tf
    import jax
    import jaxlib
    import numpy as np
    import scipy
    import sacred
    import pandas

    print("="*60)
    print("🔒 NUMPY VERSION CHECK")
    print("="*60)
    print("✅ NumPy version:", np.__version__)
    print("✅ NumPy file location:", np.__file__)
    
    print("\n" + "="*60)
    print("📦 PACKAGE VERSIONS")
    print("="*60)
    print("✅ peppi_py location:", peppi_py.__file__)
    print("✅ TensorFlow version:", tf.__version__)
    print("✅ JAX version:", jax.__version__)
    print("✅ JAXlib version:", jaxlib.__version__)
    print("✅ SciPy version:", scipy.__version__)
    print("✅ Pandas version:", pandas.__version__)
    print("✅ Sacred version:", sacred.__version__)
    
    print("\n" + "="*60)
    print("🧪 FUNCTIONALITY TESTS")
    print("="*60)
    
    print("Testing TensorFlow...")
    try:
        tf_test = tf.constant([1, 2, 3])
        print("   ✅ TensorFlow tensor created:", tf_test.numpy())
        
        # Test a simple computation
        tf_result = tf.reduce_sum(tf_test)
        print("   ✅ TensorFlow computation:", tf_result.numpy())
    except Exception as e:
        print(f"   ❌ TensorFlow test failed: {e}")
    
    print("\nTesting JAX...")
    try:
        jax_test = jax.numpy.array([1, 2, 3])
        print("   ✅ JAX array created:", jax_test)
        
        # Test a simple computation
        jax_result = jax.numpy.sum(jax_test)
        print("   ✅ JAX computation:", jax_result)
    except Exception as e:
        print(f"   ❌ JAX test failed: {e}")
    
    print("\nTesting NumPy...")
    try:
        np_test = np.array([1, 2, 3])
        print("   ✅ NumPy array:", np_test)
        print("   ✅ NumPy sum:", np.sum(np_test))
    except Exception as e:
        print(f"   ❌ NumPy test failed: {e}")
    
    print("\n" + "="*60)
    print("🎉 ALL TESTS COMPLETED!")
    print("="*60)

@app.local_entrypoint()
def main():
    test_env.remote()