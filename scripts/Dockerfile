# FINAL VERSION
# This Dockerfile uses a multi-stage, hybrid conda/pip approach,
# installing build tools via apt to ensure PATH consistency.

# Start from a Conda base image.
FROM continuumio/miniconda3

# 1. Install system-level dependencies, including git and the Rust compiler (cargo).
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl libgl1-mesa-glx libglib2.0-0 build-essential cmake ninja-build \
    libssl-dev libffi-dev pkg-config \
    cargo && \
    rm -rf /var/lib/apt/lists/*

# 2. Use conda to create our stable base environment with Python and pinned NumPy.
RUN conda create -n slippi-env -c conda-forge python=3.10 "numpy=1.24.3" pip

# 3. Activate the conda environment for all subsequent commands.
SHELL ["conda", "run", "-n", "slippi-env", "/bin/bash", "-c"]

# 4. Now, inside the activated environment, use Pip to install our explicit list of packages.
# By installing maturin with pip, we ensure it's available for peppi-py's build.
RUN pip install \
    "maturin==1.5.1" \
    "peppi-py==0.6.0" \
    "scipy==1.10.1" \
    "pandas==2.0.3" \
    "tensorflow==2.13.0" \
    "jax==0.4.13" \
    "jaxlib==0.4.13" \
    "flax==0.7.2" \
    "optax==0.1.7" \
    "dm-haiku==0.0.10" \
    "wandb==0.15.8" \
    "tensorflow-probability==0.20.1"

# 5. Clone the repository and its submodules.
RUN git clone --recurse-submodules https://github.com/vladfi1/slippi-ai.git /root/slippi-ai
WORKDIR /root/slippi-ai

# 6. Install the project's own requirements, but use --no-deps to prevent it
# from overriding our carefully selected packages.
RUN pip install --no-deps -r requirements.txt

# 7. The final editable install for the project code.
RUN pip install -e .