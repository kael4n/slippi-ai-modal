# slippi_final_working_version.py
# This version includes an explicit environment map in the subprocess call.

import modal
import subprocess
import os

PROJECT_ROOT = "/root/slippi-ai"
REPO_URL = "https://github.com/vladfi1/slippi-ai.git"

image = modal.Image.debian_slim().apt_install(
    "tzdata", "python3", "python3-pip", "python3-dev", "build-essential",
    "pkg-config", "cmake", "ninja-build", "libssl-dev", "libffi-dev", "zlib1g-dev",
    "libbz2-dev", "libreadline-dev", "libsqlite3-dev", "libncurses5-dev",
    "libncursesw5-dev", "xz-utils", "tk-dev", "libxml2-dev", "libxmlsec1-dev",
    "liblzma-dev", "git", "curl", "wget", "unzip", "software-properties-common",
    "libgl1-mesa-glx", "libglib2.0-0", "libsm6", "libxext6", "libxrender-dev", "libgomp1"
).run_commands(
    "ln -sf /usr/bin/python3 /usr/bin/python",
    "python3 -m pip install --upgrade pip setuptools wheel",
    "python3 -m pip install maturin",
    "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
).env({
    # Set the cargo path for all subsequent commands in this image
    "PATH": "/root/.cargo/bin:$PATH",
}).run_commands(
    # Install the maturin CLI tool
    "cargo install maturin",
    # Freeze NumPy FIRST
    "pip install numpy==1.24.3 --no-deps",
    # Install peppi-py using --no-build-isolation to respect the frozen NumPy
    "pip install --no-build-isolation peppi-py==0.6.0",
).pip_install([
    "scipy==1.10.1", "jax==0.4.13", "jaxlib==0.4.13", "pandas==2.0.3",
    "tensorflow==2.13.0", "flax==0.7.2", "optax==0.1.7", "dm-haiku==0.0.10",
    "dm-tree==0.1.8", "sacred==0.8.4", "pymongo==4.5.0", "matplotlib==3.7.2",
    "seaborn==0.12.2", "tqdm==4.65.0", "cloudpickle==2.2.1", "absl-py==1.4.0",
    "tensorboard==2.13.0", "gymnasium==0.28.1", "wandb==0.15.8",
    "tensorflow-probability==0.20.1", "parameterized",
]).run_commands(
    f"git clone --recurse-submodules {REPO_URL} {PROJECT_ROOT}",
).workdir(
    PROJECT_ROOT
).run_commands(
    "pip install --no-deps -r requirements.txt",
    "pip install -e .",
)

app = modal.App("slippi-ai-final-working-build", image=image)

@app.function(
    volumes={"/data": modal.Volume.from_name("slippi-ai-dataset-doesokay")},
    timeout=7200,
    gpu="any"
)
def train():
    import sys
    import numpy as np
    
    print("--- 🚀 System Validation ---")

    print(f"✅ NumPy version: {np.__version__}")
    if np.__version__ != "1.24.3":
        print(f"❌ FATAL: NumPy version is incorrect!")
        return

    try:
        import tensorflow as tf
        print(f"✅ TensorFlow version: {tf.__version__}")
        gpu_devices = tf.config.list_physical_devices('GPU')
        print(f"✅ TensorFlow is using GPU: {gpu_devices}")
    except Exception as e:
        print(f"❌ FATAL: A core library failed to import correctly: {e}")
        return

    print("\n🎉 SUCCESS! Environment is stable and ready for training.")
    print("=" * 60)
    
    replays_path = "/data/games/Ga"
    
    command = [
        "python", "-m", "slippi_ai.train",
        "with", f"replays_path={replays_path}"
    ]
    
    print(f"👟 Running training command: {' '.join(command)}\n")

    # THE FIX: Create a copy of the current environment variables and update
    # the PYTHONPATH to ensure the subprocess inherits it correctly.
    env = os.environ.copy()
    env["PYTHONPATH"] = PROJECT_ROOT + ":" + env.get("PYTHONPATH", "")

    with subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=env, # Pass the modified environment to the subprocess
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True
    ) as p:
        for line in p.stdout:
            print(line, end='')

    if p.returncode != 0:
        print(f"\n❌ Training process failed with exit code {p.returncode}")
    else:
        print("\n✅ Training process finished successfully.")

@app.local_entrypoint()
def main():
    train.remote()