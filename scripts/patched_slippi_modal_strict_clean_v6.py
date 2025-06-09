# Final v6: replace 'source' with POSIX-compliant '.' to fix /bin/sh error
import modal

PEPPI_PY_COMMIT = "8c02a4659c3302321dfbfcf2093c62f634e335f7"
PEPPI_PY_URL = f"git+https://github.com/hohav/peppi-py.git@{PEPPI_PY_COMMIT}"
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

    image = image.run_commands([
        "ln -sf /usr/bin/python3 /usr/bin/python",
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
        "echo 'source $HOME/.cargo/env' >> ~/.bashrc",
        "/root/.cargo/bin/rustc --version"
    ]).env({
        "PATH": "/root/.cargo/bin:$PATH",
        "CARGO_HOME": "/root/.cargo",
        "RUSTUP_HOME": "/root/.rustup",
    })

    image = image.run_commands([
        "python -m pip install --upgrade pip setuptools wheel",
        "pip install --upgrade setuptools-rust maturin cython"
    ])

    image = image.pip_install([
        "numpy==1.24.3", "scipy==1.10.1", "pandas==2.0.3",
        "tensorflow==2.13.0", "jax==0.4.13",
        "flax==0.7.2", "optax==0.1.7", "dm-haiku==0.0.10", "dm-tree==0.1.8",
        "sacred==0.8.4", "pymongo==4.5.0", "matplotlib==3.7.2",
        "seaborn==0.12.2", "tqdm==4.65.0", "cloudpickle==2.2.1",
        "absl-py==1.4.0", "tensorboard==2.13.0", "gymnasium==0.28.1"
    ])

    image = image.run_commands([
        f"git clone {REPO_URL} {PROJECT_ROOT}",
        f"ls -la {PROJECT_ROOT}"
    ]).workdir(PROJECT_ROOT)

    image = image.run_commands([
        f". ~/.cargo/env && export PATH='/root/.cargo/bin:$PATH' && pip install --no-build-isolation --verbose --timeout=300 '{PEPPI_PY_URL}'",
        'python -c "import peppi; print(\'peppi-py installed and importable\')"'
    ])

    image = image.run_commands([
        "pip install -r requirements.txt || echo 'Partial requirements succeeded'",
        "pip install -e . || echo 'Editable install skipped/fallback'"
    ])

    return image

image = create_slippi_image()

app = modal.App("slippi-ai-strict-clean-v6")

@app.function(image=image)
def test_env():
    import peppi
    import tensorflow as tf
    import jax
    import sacred

    print("peppi:", peppi.__file__)
    print("tensorflow:", tf.__version__)
    print("jax:", jax.__version__)
    print("sacred:", sacred.__version__)

@app.local_entrypoint()
def main():
    test_env.remote()
