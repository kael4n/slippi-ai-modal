# slippi_true_replication.py
# This script abandons the Dockerfile approach and is a high-fidelity
# recreation of the proven working script's installation logic and sequence.

import modal

PROJECT_ROOT = "/root/slippi-ai"
REPO_URL = "https://github.com/vladfi1/slippi-ai.git"

# STAGE 1: Replicate the working script's base image and system setup EXACTLY.
image = modal.Image.debian_slim().apt_install(
    "tzdata", "python3", "python3-pip", "python3-dev", "build-essential",
    "pkg-config", "cmake", "ninja-build", "libssl-dev", "libffi-dev", "zlib1g-dev",
    "libbz2-dev", "libreadline-dev", "libsqlite3-dev", "libncurses5-dev",
    "libncursesw5-dev", "xz-utils", "tk-dev", "libxml2-dev", "libxmlsec1-dev",
    "liblzma-dev", "git", "curl", "wget", "unzip", "software-properties-common",
    "libgl1-mesa-glx", "libglib2.0-0", "libsm6", "libxext6", "libxrender-dev", "libgomp1"
)

# STAGE 2: Install Rust and Maturin using the exact same commands and environment.
# This ensures they are on the correct PATH for subsequent steps.
image = image.run_commands(
    "ln -sf /usr/bin/python3 /usr/bin/python",
    "python3 -m pip install --upgrade pip setuptools wheel",
    "python3 -m pip install maturin",
    "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
    "/root/.cargo/bin/cargo install maturin",
).env({
    "PATH": "/root/.cargo/bin:$PATH",
})

# STAGE 3: Freeze NumPy FIRST. This is a critical step from the working script.
image = image.run_commands(
    "python3 -m pip install numpy==1.24.3 --no-deps"
)

# STAGE 4: Install peppi-py using --no-build-isolation to respect the frozen NumPy.
image = image.run_commands(
    "python3 -m pip install --no-build-isolation peppi-py==0.6.0",
)

# STAGE 5: Freeze all other core scientific libraries using pip_install.
# This happens BEFORE installing from requirements.txt.
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

# STAGE 6: Clone the repo with submodules.
image = image.run_commands(
    f"git clone --recurse-submodules {REPO_URL} {PROJECT_ROOT}",
).workdir(PROJECT_ROOT)

# STAGE 7: Let requirements.txt fill the gaps, which will now respect the frozen environment.
# Then, do the final editable install. This is the last build step.
image = image.run_commands(
    "python3 -m pip install -r requirements.txt",
    "python3 -m pip install -e ."
)

# --- Modal App Definition ---
app = modal.App("slippi-ai-replicated-build", image=image)

@app.function(
    volumes={"/data": modal.Volume.from_name("slippi-ai-dataset-doesokay")},
    timeout=3600,
    gpu="any"
)
def train():
    import sys
    import numpy as np
    
    print("--- 🚀 System Validation and Training ---")
    sys.path.insert(0, PROJECT_ROOT)

    print(f"✅ NumPy version: {np.__version__}")
    if np.__version__ != "1.24.3":
        print(f"❌ FATAL: NumPy version is incorrect! Expected 1.24.3, got {np.__version__}.")
        return

    try:
        import tensorflow as tf
        import jax
        import slippi_ai
        print(f"✅ TensorFlow version: {tf.__version__}")
        print(f"✅ JAX version: {jax.__version__}")
        gpu_devices = tf.config.list_physical_devices('GPU')
        print(f"✅ TensorFlow is using GPU: {gpu_devices}")
    except (ImportError, AttributeError) as e:
        print(f"❌ FATAL: A core library failed to import correctly: {e}")
        return

    print("\n🎉 SUCCESS! Environment is stable and ready for training.")

@app.local_entrypoint()
def main():
    train.remote()