import modal

# === CONFIG ===
PROJECT_ROOT = "/root/slippi-test"
PEPPI_PY_URL = "git+https://github.com/hohav/peppi-py.git@8c02a4659c3302321dfbfcf2093c62f634e335f7"

# === IMAGE BUILD ===
def build_test_image():
    base = modal.Image.debian_slim().apt_install([
        "build-essential", "curl", "git", "pkg-config",
        "libssl-dev", "libffi-dev", "cmake", "python3-dev",
        "ninja-build", "libgl1-mesa-glx", "libglib2.0-0"
    ])

    base = base.run_commands([
        "curl https://sh.rustup.rs -sSf | sh -s -- -y",
        "echo 'source $HOME/.cargo/env' >> ~/.bashrc"
    ]).env({
        "PATH": "/root/.cargo/bin:$PATH",
        "CARGO_HOME": "/root/.cargo",
        "RUSTUP_HOME": "/root/.rustup",
    })

    base = base.run_commands([
        "python3 -m pip install --upgrade pip setuptools wheel",
        "pip install setuptools-rust maturin cython",

        # Install JAX and jaxlib CUDA via correct index
        "pip install --upgrade "
        "--find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html "
        "jax==0.4.13 jaxlib==0.4.13+cuda12.cudnn89"
    ])

    base = base.pip_install([
        "tensorflow==2.13.0",
        "sacred==0.8.4",
        "gym==0.26.2",
        "gymnasium==0.28.1"
    ])

    # ✅ peppi-py installation - use run_commands for custom pip flags
    base = base.run_commands([
        f"pip install --no-build-isolation 'peppi-py @ {PEPPI_PY_URL}'"
    ])

    return base

# === BUILD IMAGE ===
image = build_test_image()

# === TEST FUNCTION ===
app = modal.App("slippi-ai-test-lite")

@app.function(
    image=image,
    gpu="A10G",
    memory=8192,
    timeout=600
)
def validate_test_env():
    print("🔍 Running minimal validation...\n")
    results = {"success": [], "fail": []}

    def check(name, fn):
        try:
            fn()
            print(f"✅ {name}")
            results["success"].append(name)
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            results["fail"].append(f"{name}: {e}")

    check("peppi", lambda: __import__("peppi"))
    check("tensorflow", lambda: __import__("tensorflow"))
    check("jax", lambda: __import__("jax"))
    check("sacred", lambda: __import__("sacred"))

    # TensorFlow GPU check
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        print(f"🖥️ TensorFlow GPU devices: {gpus}")
    except Exception as e:
        print(f"⚠️ TensorFlow GPU check failed: {e}")

    # JAX device check
    try:
        import jax
        print(f"🔧 JAX devices: {jax.devices()}")
    except Exception as e:
        print(f"⚠️ JAX device check failed: {e}")

    print("\n✅ Passed:", results["success"])
    print("❌ Failed:", results["fail"])

# === ENTRYPOINT ===
@app.local_entrypoint()
def main():
    print("🚀 Launching test")
    result = validate_test_env.remote()
    if result:
        result.get()
    else:
        print("❌ Failed to launch remote function")