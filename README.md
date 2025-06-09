# Slippi AI on Modal

[cite_start]This is a fork of Vlad Fiacco's (vladfi1) [slippi-ai project](https://github.com/vladfi1/slippi-ai)[cite: 55], adapted to run training and experiments on the [Modal Labs](https://modal.com/) cloud compute platform.

## Project Goal

The primary goal of this repository is to leverage Modal for handling the complex Rust-based dependencies (like `peppi-py`) and for scaling up AI model training on GPUs in the cloud.

## Structure

The original project structure has been reorganized for clarity:

* [cite_start]`Dockerfile.slippi`: A Dockerfile for containerizing the project.
* [cite_start]`/scripts/working/`: Contains the main, active development scripts.
    * [cite_start]`experimental.py`: The current script for ongoing experiments.
    * [cite_start]Other stable or working scripts are also located here.
* [cite_start]`/scripts/archive/`: Contains older or deprecated script versions for historical reference.
* [cite_start]`requirements.txt`: The specific Python dependencies for the project.
* [cite_start]`setup.py`: The project's installation script.

## How to Run

This project is intended to be run via Modal.

1.  **Prerequisites**: Ensure you have Python and the Modal client installed (`pip install modal-client`).
2.  **Configuration**: Create a `.env` file for any necessary API keys (e.g., for `wandb`).
3.  **Execution**: Run the desired script using the Modal CLI. For example:
    ```bash
    modal run scripts/working/experimental.py
    ```