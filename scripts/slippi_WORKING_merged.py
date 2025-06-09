# slippi_WORKING_merged.py
# Combined environment + .pkl validation runner using a single Modal app

import modal
import pickle
from pathlib import Path
import sys

PEPPI_PY_URL = "git+https://github.com/hohav/peppi-py.git@8c02a4659c3302321dfbfcf2093c62f634e335f7"
PROJECT_ROOT = "/root/slippi-ai"
REPO_URL = "https://github.com/vladfi1/slippi-ai.git"
PKL_DIR = "/dataset/games/Ga"
VOLUME_NAME = "slippi-ai-dataset-doesokay"

def create_slippi_image():
    base_image = modal.Image.from_registry(
        "ubuntu:22.04",
        setup_dockerfile_commands=[
            "ENV DEBIAN_FRONTEND=noninteractive",
            "RUN apt-get update && apt-get install -y tzdata python3 python3-pip python3-dev",
            "RUN ln -sf /usr/bin/python3 /usr/bin/python",
        ]
    )

    image = base_image.apt_install([
        "build-essential", "pkg-config", "cmake", "ninja-build",
        "libssl-dev", "libffi-dev", "zlib1g-dev", "libbz2-dev",
        "libreadline-dev", "libsqlite3-dev", "libncurses5-dev",
        "libncursesw5-dev", "xz-utils", "tk-dev", "libxml2-dev",
        "libxmlsec1-dev", "liblzma-dev", "git", "curl", "wget",
        "unzip", "software-properties-common", "libgl1-mesa-glx",
        "libglib2.0-0", "libsm6", "libxext6", "libxrender-dev",
        "libgomp1",
    ])

    image = image.run_commands([
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
        "echo 'source $HOME/.cargo/env' >> ~/.bashrc",
        "/root/.cargo/bin/rustc --version",
    ]).env({
        "PATH": "/root/.cargo/bin:$PATH",
        "CARGO_HOME": "/root/.cargo",
        "RUSTUP_HOME": "/root/.rustup",
    })

    image = image.run_commands([
        "python -m pip install --upgrade pip setuptools wheel",
        "pip install --upgrade setuptools-rust maturin cython",
    ])

    image = image.pip_install([
        "numpy==1.24.3", "scipy==1.10.1", "pandas==2.0.3",
        "tensorflow==2.13.0", "jax[cuda12_pip]==0.4.13", "jaxlib==0.4.13",
        "flax==0.7.2", "optax==0.1.7", "dm-haiku==0.0.10", "dm-tree==0.1.8",
        "sacred==0.8.4", "pymongo==4.5.0", "matplotlib==3.7.2",
        "seaborn==0.12.2", "tqdm==4.65.0", "cloudpickle==2.2.1",
        "absl-py==1.4.0", "tensorboard==2.13.0", "gymnasium==0.28.1",
        "gym==0.21.0"
    ])

    image = image.run_commands([
        f"git clone {REPO_URL} {PROJECT_ROOT}",
        f"ls -la {PROJECT_ROOT}"
    ]).workdir(PROJECT_ROOT)

    image = image.run_commands([
        f"""
echo '=== Installing peppi-py from fixed commit ==='
source ~/.cargo/env
export PATH="/root/.cargo/bin:$PATH"

pip install --no-build-isolation --verbose --timeout=300 "{PEPPI_PY_URL}"
python -c "import peppi; print('✅ peppi-py installed and importable')"
        """
    ])

    image = image.run_commands([
        "pip install -r requirements.txt || echo 'Some requirements may have failed - continuing'",
        "pip install -e . || echo 'Project installation deferred'",
    ])

    return image

image = create_slippi_image()
volume = modal.Volume.from_name(VOLUME_NAME)
app = modal.App("slippi-ai-merged")

@app.function(image=image)
def test_env():
    import peppi
    import tensorflow as tf
    import jax
    import sacred
    print("✅ peppi", peppi.__file__)
    print("✅ tensorflow", tf.__version__)
    print("✅ jax", jax.__version__)
    print("✅ sacred", sacred.__version__)

@app.function(image=image, volumes={"/dataset": volume})
def validate_pickles():
    print(f"🔍 Scanning {PKL_DIR} for .pkl files...")
    path = Path(PKL_DIR)
    files = list(path.glob("*.pkl"))

    print(f"📦 Found {len(files)} .pkl files")
    if not files:
        print("⚠️ No pickle files found.")
        return

    for file in files[:5]:
        print(f"📂 Checking: {file.name}")
        try:
            with open(file, "rb") as f:
                data = pickle.load(f)
                print(f"   ✅ Loaded. Type: {type(data)}")
                if isinstance(data, dict):
                    print(f"   🔑 Keys: {list(data.keys())[:5]}")
                elif isinstance(data, list):
                    print(f"   📏 Length: {len(data)}")
        except Exception as e:
            print(f"   ❌ Failed to load {file.name}: {e}")

@app.local_entrypoint()
def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "env"
    if mode == "env":
        test_env.remote()
    elif mode == "pkl":
        validate_pickles.remote()
    else:
        print(f"⚠️ Unknown mode: {mode} (use 'env' or 'pkl')")
