import modal

# === CONFIG ===
PROJECT_ROOT = "/root/slippi-test"

# === ROBUST IMAGE BUILD ===
def build_test_image():
    # Start with Ubuntu 22.04 for better compatibility
    base = modal.Image.from_registry(
        "ubuntu:22.04",
        setup_dockerfile_commands=[
            "ENV DEBIAN_FRONTEND=noninteractive",
            "ENV TZ=UTC",
            "RUN apt-get update && apt-get install -y python3 python3-pip python3-dev tzdata",
            "RUN ln -fs /usr/share/zoneinfo/UTC /etc/localtime", 
            "RUN dpkg-reconfigure --frontend noninteractive tzdata",
            "RUN ln -s /usr/bin/python3 /usr/bin/python",
        ]
    )

    # Install comprehensive system dependencies
    base = base.apt_install([
        # Build essentials
        "build-essential", "pkg-config", "cmake", "ninja-build",
        # System libraries  
        "libssl-dev", "libffi-dev", "zlib1g-dev", "libbz2-dev",
        "libreadline-dev", "libsqlite3-dev", "libncurses5-dev",
        "libncursesw5-dev", "xz-utils", "tk-dev", "libxml2-dev",
        "libxmlsec1-dev", "liblzma-dev",
        # Git and utilities
        "git", "curl", "wget", "unzip", "software-properties-common",
        # Graphics libraries for TensorFlow
        "libgl1-mesa-glx", "libglib2.0-0", "libsm6", "libxext6",
        "libxrender-dev", "libgomp1",
    ])

    # Install Rust with explicit environment setup
    base = base.run_commands([
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable",
        "echo 'source $HOME/.cargo/env' >> ~/.bashrc",
    ]).env({
        "PATH": "/root/.cargo/bin:$PATH",
        "CARGO_HOME": "/root/.cargo", 
        "RUSTUP_HOME": "/root/.rustup",
    })

    # Verify Rust installation
    base = base.run_commands([
        "bash -c '. ~/.cargo/env && rustc --version'",
        "bash -c '. ~/.cargo/env && cargo --version'",
    ])

    # Upgrade pip and install build tools (including maturin)
    base = base.run_commands([
        "python -m pip install --upgrade pip setuptools wheel",
        "pip install --upgrade setuptools-rust maturin cython",
    ])

    # Install core dependencies first
    base = base.pip_install([
        "numpy==1.24.3",
        "scipy==1.10.1", 
        "pandas==2.0.3",
    ])

    # Install TensorFlow and JAX
    base = base.run_commands([
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

    # FIXED: Use the working peppi-py installation method from debug
    base = base.run_commands([
        # Ensure Rust environment is available and maturin is installed
        "bash -c 'source ~/.cargo/env && export PATH=\"/root/.cargo/bin:$PATH\" && rustc --version && cargo --version'",
        
        # Use the WORKING method: build from source with no-binary flag
        "bash -c '"
        "source ~/.cargo/env && "
        "export PATH=\"/root/.cargo/bin:$PATH\" && "
        "echo \"Installing peppi-py from source (working method)...\" && "
        "pip install git+https://github.com/hohav/peppi-py.git --no-binary=peppi-py --force-reinstall && "
        "echo \"✅ peppi-py source build completed\"'",
        
        # Verify installation with CORRECT import syntax
        "python -c 'from peppi_py import read_slippi, read_peppi; print(\"✅ peppi_py successfully installed and importable\")' || "
        "echo '❌ peppi_py installation verification failed'"
    ])

    # Set environment variables
    base = base.env({
        "PYTHONPATH": PROJECT_ROOT,
        "TF_CPP_MIN_LOG_LEVEL": "1",
        "TF_ENABLE_ONEDNN_OPTS": "0", 
        "CUDA_VISIBLE_DEVICES": "0",
        "TF_FORCE_GPU_ALLOW_GROWTH": "true",
    })

    return base

# === BUILD IMAGE ===
image = build_test_image()

# === TEST FUNCTION ===
app = modal.App("slippi-ai-test-robust")

@app.function(
    image=image,
    gpu="A10G",
    memory=8192,
    timeout=600
)
def validate_test_env():
    print("🔍 Running comprehensive validation...\n")
    results = {"success": [], "fail": [], "warnings": []}

    def check(name, fn, critical=True):
        try:
            result = fn()
            print(f"✅ {name}")
            if result:
                print(f"   Details: {result}")
            results["success"].append(name)
            return True
        except Exception as e:
            error_msg = f"{name}: {e}"
            if critical:
                print(f"❌ {error_msg}")
                results["fail"].append(error_msg)
            else:
                print(f"⚠️ {error_msg}")
                results["warnings"].append(error_msg)
            return False

    # FIXED: Use correct import syntax for peppi_py
    def peppi_check():
        from peppi_py import read_slippi, read_peppi
        return "peppi_py imported with correct syntax"

    check("peppi_py", peppi_check)
    check("tensorflow", lambda: f"Version {__import__('tensorflow').__version__}")
    check("jax", lambda: f"Version {__import__('jax').__version__}")
    check("sacred", lambda: f"Version {__import__('sacred').__version__}")

    # Detailed checks
    def tf_gpu_check():
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            # Configure GPU memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            return f"Found {len(gpus)} GPU(s): {gpus}"
        else:
            return "No GPUs found - will use CPU"
    
    def jax_device_check():
        import jax
        devices = jax.devices()
        return f"JAX devices: {devices}"
    
    def peppi_functionality_check():
        from peppi_py import read_slippi, read_peppi
        # Verify functions are callable
        if callable(read_slippi) and callable(read_peppi):
            return f"peppi_py functions read_slippi and read_peppi are available and callable"
        else:
            raise Exception("peppi_py functions are not callable")

    check("TensorFlow GPU", tf_gpu_check, critical=False)
    check("JAX devices", jax_device_check, critical=False)
    check("peppi_py functionality", peppi_functionality_check, critical=True)

    # System info
    def system_info():
        import sys
        import platform
        return f"Python {sys.version} on {platform.platform()}"
    
    check("System Info", system_info, critical=False)

    # Summary
    print(f"\n📊 === VALIDATION SUMMARY ===")
    print(f"✅ Successes: {len(results['success'])}")
    print(f"⚠️ Warnings: {len(results['warnings'])}")  
    print(f"❌ Critical Failures: {len(results['fail'])}")

    if results["fail"]:
        print(f"\n❌ CRITICAL FAILURES:")
        for failure in results["fail"]:
            print(f"  - {failure}")
    
    if results["warnings"]:
        print(f"\n⚠️ WARNINGS:")
        for warning in results["warnings"]:
            print(f"  - {warning}")
    
    if results["success"]:
        print(f"\n✅ SUCCESSES:")
        for success in results["success"]:
            print(f"  - {success}")

    # Overall status
    if len(results["fail"]) == 0:
        print(f"\n🎉 Environment validation PASSED!")
        return {"status": "success", "results": results}
    else:
        print(f"\n💥 Environment validation FAILED due to critical failures!")
        return {"status": "failed", "results": results}

@app.function(
    image=image,
    gpu="A10G", 
    memory=4096,
    timeout=300
)
def quick_peppi_test():
    """Quick test specifically for peppi_py functionality with CORRECT import"""
    print("🧪 === PEPPI_PY SPECIFIC TEST ===")
    
    try:
        # FIXED: Use correct import syntax
        from peppi_py import read_slippi, read_peppi
        print("✅ peppi_py module imported successfully with correct syntax")
        
        # Check if functions are callable
        if callable(read_slippi):
            print("✅ read_slippi function is available and callable")
        else:
            print("❌ read_slippi function is not callable")
            
        if callable(read_peppi):
            print("✅ read_peppi function is available and callable")
        else:
            print("❌ read_peppi function is not callable")
        
        # Try to check version if available
        try:
            import peppi_py
            if hasattr(peppi_py, '__version__'):
                print(f"✅ peppi_py version: {peppi_py.__version__}")
            else:
                print("ℹ️ peppi_py version not available")
        except:
            print("ℹ️ peppi_py version check failed")
            
        return {"success": True, "message": "peppi_py is working with correct import syntax"}
        
    except ImportError as e:
        print(f"❌ Cannot import peppi_py: {e}")
        return {"success": False, "error": f"Import error: {e}"}
    except Exception as e:
        print(f"❌ peppi_py test failed: {e}")
        return {"success": False, "error": f"Test error: {e}"}

# === DEBUG FUNCTION FOR TROUBLESHOOTING ===
@app.function(
    image=image,
    gpu="A10G",
    memory=4096,
    timeout=300
)
def debug_peppi_installation():
    """Debug function to verify peppi_py installation"""
    print("🔍 === DEBUGGING PEPPI_PY INSTALLATION ===")
    
    import subprocess
    import sys
    import os
    
    # Check Rust installation
    print("\n--- Rust Installation Check ---")
    try:
        result = subprocess.run(['rustc', '--version'], capture_output=True, text=True)
        print(f"✅ Rust version: {result.stdout.strip()}")
    except Exception as e:
        print(f"❌ Rust not found: {e}")
    
    try:
        result = subprocess.run(['cargo', '--version'], capture_output=True, text=True)
        print(f"✅ Cargo version: {result.stdout.strip()}")
    except Exception as e:
        print(f"❌ Cargo not found: {e}")
    
    # Check pip packages
    print("\n--- Pip Packages Check ---")
    result = subprocess.run([sys.executable, '-m', 'pip', 'list'], capture_output=True, text=True)
    pip_packages = result.stdout
    
    if 'peppi' in pip_packages.lower():
        print("✅ Found peppi-related packages:")
        for line in pip_packages.split('\n'):
            if 'peppi' in line.lower():
                print(f"  {line}")
    else:
        print("❌ No peppi packages found in pip list")
    
    # Test CORRECT import syntax
    print("\n--- Import Test (Correct Syntax) ---")
    try:
        from peppi_py import read_slippi, read_peppi
        print("✅ Successfully imported peppi_py with correct syntax")
        print(f"✅ read_slippi callable: {callable(read_slippi)}")
        print(f"✅ read_peppi callable: {callable(read_peppi)}")
    except ImportError as e:
        print(f"❌ Cannot import peppi_py: {e}")
    except Exception as e:
        print(f"❌ Import test failed: {e}")
    
    # Check environment variables
    print("\n--- Environment Variables ---")
    print(f"PATH: {os.environ.get('PATH', 'Not set')}")
    print(f"CARGO_HOME: {os.environ.get('CARGO_HOME', 'Not set')}")
    print(f"RUSTUP_HOME: {os.environ.get('RUSTUP_HOME', 'Not set')}")
    
    return {"debug_complete": True}

# === ENTRYPOINT ===
@app.local_entrypoint()
def main():
    print("🚀 Launching robust validation tests with FIXED peppi_py installation")
    
    # First, run debug to verify installation
    print("\n=== Running debug analysis ===")
    try:
        debug_result = debug_peppi_installation.remote()
        print("Debug analysis completed")
    except Exception as e:
        print(f"Debug analysis failed: {e}")
    
    print("\n=== Running comprehensive validation ===")
    try:
        validation_result = validate_test_env.remote()
        print("Comprehensive validation completed")
        
        # Extract status from the returned dictionary
        if isinstance(validation_result, dict):
            validation_status = validation_result.get('status', 'unknown')
        else:
            validation_status = 'unknown'
            
    except Exception as e:
        print(f"Validation failed: {e}")
        validation_status = 'failed'
    
    print(f"\n=== Running peppi_py-specific test ===")
    try:
        peppi_test_result = quick_peppi_test.remote()
        print("Peppi-specific test completed")
        
        # Extract success from the returned dictionary
        if isinstance(peppi_test_result, dict):
            peppi_success = peppi_test_result.get('success', False)
            peppi_error = peppi_test_result.get('error', 'unknown')
        else:
            peppi_success = False
            peppi_error = 'unknown result format'
            
    except Exception as e:
        print(f"Peppi test failed: {e}")
        peppi_success = False
        peppi_error = str(e)
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"Validation Status: {validation_status}")
    print(f"Peppi Test: {'PASSED' if peppi_success else 'FAILED'}")
    
    if not peppi_success:
        print(f"Peppi Error: {peppi_error}")
    
    overall_success = (validation_status == 'success' and peppi_success)
    
    if overall_success:
        print("🎉 ALL TESTS PASSED - Environment is ready!")
        print("\n💡 IMPORTANT: Use 'from peppi_py import read_slippi, read_peppi' in your code")
    else:
        print("💥 SOME TESTS FAILED - Check logs above for details")
        
    return overall_success