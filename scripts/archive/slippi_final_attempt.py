# slippi_final_attempt.py
import modal
from pathlib import Path

# Construct the full, absolute path to the Dockerfile
dockerfile_path = Path(__file__).parent / "Dockerfile"

image = modal.Image.from_dockerfile(dockerfile_path)

app = modal.App("slippi-ai-final-build", image=image)

@app.function(
    volumes={"/data": modal.Volume.from_name("slippi-ai-dataset-doesokay")},
    timeout=3600,
    gpu="any"
)
def train():
    import sys
    import numpy as np
    
    print("--- 🚀 System Validation and Training ---")
    sys.path.insert(0, "/root/slippi-ai")

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