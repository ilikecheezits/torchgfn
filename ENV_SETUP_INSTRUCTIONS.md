# Environment Setup for oracle.py

This guide details how to set up a `conda` environment to run `oracle.py`, specifically targeting `arm64` architecture (e.g., Apple Silicon Macs) with CPU support.

## Prerequisites

*   **Conda:** Ensure you have Miniconda or Anaconda installed. You can download it from [conda.io](https://docs.conda.io/en/latest/miniconda.html).
*   **Git:** Make sure Git is installed to clone repositories.

## Setup Steps

1.  **Create the Conda Environment:**
    Open your terminal and run the following commands to create a new `conda` environment named `gfn-arm64` with Python 3.10 and necessary core packages:

    ```bash
    conda create -n gfn-arm64 python=3.10 -y
    conda activate gfn-arm64
    ```

2.  **Install Core Libraries (JAX, JAXlib, Flax):**
    Install `jax`, `jaxlib`, and `flax` from the `conda-forge` channel, which provides `arm64` compatible builds for CPU.

    ```bash
    conda install -c conda-forge jax jaxlib flax -y
    ```
    *Note: This will install the CPU-only version of `jaxlib` suitable for `arm64`.*

3.  **Clone `jax-rnafold`:**
    The `oracle.py` script depends on `jax-rnafold`. You need to clone this repository into your `torchgfn` project directory. Ensure you are in the `torchgfn` root directory before running this command.

    ```bash
    # Assuming you are in the /Users/Samuel/torchgfn directory
    git clone https://github.com/rkruegs123/jax-rnafold jax-rnafold
    ```
    *Note: Replace `https://github.com/your-username/jax-rnafold.git` with the actual URL of the `jax-rnafold` repository if it's different.*

4.  **Install `jax-rnafold`:**
    Navigate into the cloned `jax-rnafold` directory and install it in "editable" mode. This allows `oracle.py` to correctly import modules from `jax-rnafold`.

    ```bash
    cd jax-rnafold
    pip install -e .
    cd .. # Go back to the torchgfn root directory
    ```

5.  **Verify Installation:**
    You can test if the environment is set up correctly by trying to import `jax` and `jax_rnafold` within the activated environment:

    ```bash
    python -c "import jax; import jax_rnafold; print('JAX and jax_rnafold imported successfully!')"
    ```
    If no errors occur, the installation was successful.

## Running `oracle.py`

Once the environment is set up, you can run `oracle.py` using the following command from the `torchgfn` root directory:

```bash
conda run -n gfn-arm64 python oracle.py
PYTHONPATH=src pytest testing/test_constraints.py
```

This command activates the `gfn-arm64` environment and then executes the `oracle.py` script.

## Installing torchgfn

To install the main `torchgfn` library and its dependencies (as defined in `pyproject.toml`), navigate back to the `torchgfn` root directory (if you're not already there) and run:

```bash
pip install -e .
```

### Troubleshooting `torchgfn` installation

If you encounter issues, particularly `No matching distribution found` errors for packages like `torch` or `tensordict`, consider the following:

*   **Python Version**: Ensure your Python version (e.g., 3.10 as specified for this `conda` environment) is compatible with the required package versions.
*   **pip Version**: Ensure your `pip` is up-to-date (`pip install --upgrade pip`).
*   **Dependency Conflicts**: Sometimes, conflicts with pre-installed packages in your `conda` environment can occur. If persistent errors arise, consider creating a fresh `conda` environment for `torchgfn` and installing its dependencies there.

