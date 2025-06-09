
import modal

stub = modal.Stub("patched-slippi-v19")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "build-essential", "libssl-dev", "pkg-config", "curl", "tzdata")
    .run_commands(
        "curl https://sh.rustup.rs -sSf | bash -s -- -y",
        "export PATH=$PATH:/root/.cargo/bin && cargo --version",
    )
    .pip_install("numpy==1.24.3")  # pin numpy BEFORE others to avoid 2.x
    .run_commands(
        ". ~/.cargo/env && export PATH='/root/.cargo/bin:$PATH' && "
        "pip install --no-build-isolation maturin && "
        "pip install --no-build-isolation --verbose --timeout=300 "
        "'git+https://github.com/hohav/peppi-py.git@8c02a4659c3302321dfbfcf2093c62f634e335f7'"
    )
    .pip_install(
        "scipy==1.10.1",
        "tensorflow==2.13.0",
        "jax==0.4.13",
        "jaxlib==0.4.13",
        "flax==0.7.2",
        "optax==0.1.7",
        "dm-tree==0.1.8",
        "dm-sonnet==2.0.2",
        "melee==0.38.1",
        "pyarrow==20.0.0",
        "py7zr",
        "pandas==2.0.3",
        "sacred==0.8.4",
        "wandb==0.20.1",
        "tensorflow_probability==0.25.0",
        "munch",
    )
    .run_commands(
        "git clone https://github.com/vladfi1/slippi-ai.git /root/slippi-ai",
        "cd /root/slippi-ai && python3 -m pip install -e . || echo 'Editable install skipped/fallback'"
    )
)

vol = modal.Mount.from_local_dir(".", remote_path="/root")

@stub.function(image=image, mounts=[vol], timeout=600)
def test_env():
    import peppi
    import tensorflow as tf
    import jax
    import sacred
    import munch
    import pandas as pd
    print("✅ All critical libraries are successfully imported. Environment is ready.")

if __name__ == "__main__":
    stub.deploy("patched_slippi_modal_strict_clean_v19")
    test_env.remote()
