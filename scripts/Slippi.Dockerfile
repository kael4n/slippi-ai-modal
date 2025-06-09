# FINAL DOCKERFILE - SLIPPI AI
# This version uses a hyper-sequential, multi-stage install process
# combining Conda for the base and Pip for individual components.

# Start from a Conda base image.
FROM continuumio/miniconda3

# 1. Install system-level tools.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl libgl1-mesa-glx libglib2.0-0 build-essential cmake ninja-build \
    libssl-dev libffi-dev pkg-config \
    # Add Rust's compiler directly from apt.
    cargo && \
    rm -rf /var/lib/apt/lists/*

# 2. Use conda to create a new environment with ONLY Python and our pinned NumPy.
# This creates a stable, isolated foundation.
RUN conda create -n slippi-env -c conda-forge \
    python=3.10 \
    "numpy=1.24.3" \
    pip

# 3. Activate the conda environment for all subsequent commands.
# This is the most important step for ensuring consistency.
SHELL ["conda", "run", "-n", "slippi-env", "/bin/bash", "-c"]

# 4. Now, inside the activated environment, install packages with Pip one by one
# or in small, logical groups.

# First, the build tool for peppi-py.
RUN pip install "maturin==1.5.1"

# Second, the library that depends on the build tool.
RUN pip install "peppi-py==0.6.0"

# Third, the core ML stack. These are known to work together.
RUN pip install \
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

# 6. Install the project's own requirements from its requirements.txt file.
RUN pip install -r requirements.txt

# 7. The final editable install for the project code.
RUN pip install -e .