# slippi_compatible_solution.py
import modal
from pathlib import Path

# Construct the full, absolute path to the Dockerfile
# This ensures the script can always find it.
dockerfile_path = Path(__file__).parent / "Slippi.Dockerfile"

# This points to the Dockerfile using its full path.
image = modal.Image.from_dockerfile(dockerfile_path)

app = modal.App("slippi-ai-final-build", image=image)

@app.function(
    volumes={"/data": modal.Volume.from_name("slippi-ai-dataset-doesokay")},
    timeout=3600,
    gpu="any"
)
def train():
    import os
    import sys
    import numpy as np
    import tensorflow as tf

    print("--- 🚀 System Validation and Training ---")
    sys.path.insert(0, "/root/slippi-ai")

    print(f"✅ NumPy version: {np.__version__}")
    if np.__version__ != "1.24.3":
        print("❌ NumPy version is incorrect!")
        return
        
    print(f"✅ TensorFlow version: {tf.__version__}")
    if not tf.config.list_physical_devices('GPU'):
         print("⚠️ WARNING: No GPU detected by TensorFlow.")

    print("\n🎉 SUCCESS! Environment is stable.")

@app.local_entrypoint()
def main():
    train.remote()