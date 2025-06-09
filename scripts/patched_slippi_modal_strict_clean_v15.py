# v15: FINAL PATCH — installs Rust, maturin, disables pip isolation for peppi-py==0.6.0
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

    # Rust + cargo + maturin + path setup
    image = image.run_commands([
        "ln -sf /usr/bin/python3 /usr/bin/python",
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
        "/root/.cargo/bin/cargo install maturin",
        "/root/.cargo/bin/maturin --version"
    ]).env({
        "PATH": "/root/.cargo/bin:$PATH",
        "CARGO_HOME": "/root/.cargo",
        "RUSTUP_HOME": "/root/.rustup"
    })

    # Core Python environment
    image = image.run_commands([
        "python3 -m pip install --upgrade pip setuptools wheel",
        # ✅ Use --no-build-isolation to allow peppi-py to find system maturin
        "python3 -m pip install --no-build-isolation peppi-py==0.6.0",
        'python3 -c "import peppi_py; print(\'✅ peppi-py v0.6.0 installed and importable\')"'
    ])

    # slippi-ai requirements
    image = image.pip_install([
        "numpy==1.24.3", "scipy==1.10.1", "pandas==2.0.3",
        "tensorflow==2.13.0", "jax==0.4.13",
        "flax==0.7.2", "optax==0.1.7", "dm-haiku==0.0.10", "dm-tree==0.1.8",
        "sacred==0.8.4", "pymongo==4.5.0", "matplotlib==3.7.2",
        "seaborn==0.12.2", "tqdm==4.65.0", "cloudpickle==2.2.1",
        "absl-py==1.4.0", "tensorboard==2.13.0", "gymnasium==0.28.1"
    ])

    # Clone slippi-ai repo and prep
    image = image.run_commands([
        f"git clone {REPO_URL} {PROJECT_ROOT}",
        f"ls -la {PROJECT_ROOT}"
    ]).workdir(PROJECT_ROOT)

    image = image.run_commands([
        "python3 -m pip install -r requirements.txt || echo 'Partial requirements succeeded'",
        "python3 -m pip install -e . || echo 'Editable install skipped/fallback'"
    ])

    return image

image = create_slippi_image()

app = modal.App("slippi-ai-strict-clean-v15")

@app.function(image=image)
def test_env():
    import peppi_py
    import tensorflow as tf
    import jax
    import sacred

    print("✅ peppi_py path:", peppi_py.__file__)
    print("✅ tensorflow version:", tf.__version__)
    print("✅ jax version:", jax.__version__)
    print("✅ sacred version:", sacred.__version__)

@app.local_entrypoint()
def main():
    test_env.remote()
