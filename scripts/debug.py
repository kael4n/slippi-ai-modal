import modal

# === FIXED PEPPI-PY DEBUG SCRIPT ===

def build_peppi_debug_image():
    """Fixed image build with proper Rust installation"""
    
    base = modal.Image.from_registry(
        "ubuntu:22.04",
        setup_dockerfile_commands=[
            "ENV DEBIAN_FRONTEND=noninteractive",
            "ENV TZ=UTC",
            "RUN apt-get update && apt-get install -y python3 python3-pip python3-dev tzdata curl git",
            "RUN ln -s /usr/bin/python3 /usr/bin/python",
        ]
    )

    # Install essential build dependencies
    base = base.apt_install([
        "build-essential", "pkg-config", "cmake", 
        "libssl-dev", "libffi-dev", "zlib1g-dev",
        "git", "curl", "wget"
    ])

    # FIXED: Install Rust properly with bash
    base = base.run_commands([
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable",
        "echo 'source $HOME/.cargo/env' >> ~/.bashrc",
        # Use bash explicitly instead of source
        "bash -c 'source ~/.cargo/env && rustc --version'",
        "bash -c 'source ~/.cargo/env && cargo --version'",
    ]).env({
        "PATH": "/root/.cargo/bin:$PATH",
        "CARGO_HOME": "/root/.cargo", 
        "RUSTUP_HOME": "/root/.rustup",
    })

    # Upgrade pip
    base = base.run_commands([
        "python -m pip install --upgrade pip setuptools wheel setuptools-rust",
    ])

    # Try different peppi-py installation strategies with proper bash sourcing
    base = base.run_commands([
        "echo '=== PEPPI-PY INSTALLATION ATTEMPT ==='",
        
        # Method 1: Direct from git with specific commit
        "bash -c '"
        "source ~/.cargo/env && "
        "export PATH=\"/root/.cargo/bin:$PATH\" && "
        "echo \"Method 1: Installing from git with specific commit...\" && "
        "pip install --no-build-isolation --verbose "
        "\"git+https://github.com/hohav/peppi-py.git@8c02a4659c3302321dfbfcf2093c62f634e335f7\" "
        "2>&1 | tee /tmp/peppi_install_method1.log || echo \"Method 1 failed\"'",
        
        # Method 2: Try latest from git
        "bash -c '"
        "source ~/.cargo/env && "
        "export PATH=\"/root/.cargo/bin:$PATH\" && "
        "echo \"Method 2: Installing latest from git...\" && "
        "pip install --no-build-isolation --verbose "
        "\"git+https://github.com/hohav/peppi-py.git\" "
        "2>&1 | tee /tmp/peppi_install_method2.log || echo \"Method 2 failed\"'",
        
        # Method 3: Try from PyPI if available
        "bash -c '"
        "source ~/.cargo/env && "
        "export PATH=\"/root/.cargo/bin:$PATH\" && "
        "echo \"Method 3: Installing from PyPI...\" && "
        "pip install --verbose peppi-py "
        "2>&1 | tee /tmp/peppi_install_method3.log || echo \"Method 3 failed\"'",
        
        # Verify installation
        "python -c 'import peppi; print(\"SUCCESS: peppi-py installed and importable\")' || "
        "echo 'FAILED: peppi-py not importable after installation attempts'"
    ])

    return base

app = modal.App("peppi-debug-fixed")
image = build_peppi_debug_image()

@app.function(
    image=image,
    gpu=None,
    memory=4096,
    timeout=900
)
def debug_peppi_comprehensive():
    """Comprehensive peppi-py installation debugging"""
    
    import subprocess
    import sys
    import os
    
    print("🔍 === COMPREHENSIVE PEPPI-PY DEBUG ===\n")
    
    # 1. Environment Check
    print("1. Environment Variables:")
    env_vars = ['PATH', 'CARGO_HOME', 'RUSTUP_HOME', 'PYTHONPATH']
    for var in env_vars:
        print(f"   {var}: {os.environ.get(var, 'Not set')}")
    
    # 2. Rust Installation Check
    print("\n2. Rust Installation:")
    try:
        result = subprocess.run(['rustc', '--version'], capture_output=True, text=True)
        print(f"   ✅ Rust: {result.stdout.strip()}")
        
        result = subprocess.run(['cargo', '--version'], capture_output=True, text=True)
        print(f"   ✅ Cargo: {result.stdout.strip()}")
    except Exception as e:
        print(f"   ❌ Rust/Cargo error: {e}")
    
    # 3. Python and Pip Info
    print(f"\n3. Python Environment:")
    print(f"   Python: {sys.version}")
    print(f"   Python executable: {sys.executable}")
    
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', '--version'], capture_output=True, text=True)
        print(f"   Pip: {result.stdout.strip()}")
    except Exception as e:
        print(f"   ❌ Pip error: {e}")
    
    # 4. Check if peppi-py is already installed
    print(f"\n4. Current Package Status:")
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', 'list'], capture_output=True, text=True)
        if 'peppi' in result.stdout.lower():
            print("   ✅ Found peppi-related packages:")
            for line in result.stdout.split('\n'):
                if 'peppi' in line.lower():
                    print(f"     {line}")
        else:
            print("   ❌ No peppi packages found")
    except Exception as e:
        print(f"   ❌ Package list error: {e}")
    
    # 5. Try importing peppi
    print(f"\n5. Import Test:")
    try:
        import peppi
        print("   ✅ peppi imported successfully!")
        print(f"   ✅ peppi attributes: {len([a for a in dir(peppi) if not a.startswith('_')])}")
        if hasattr(peppi, '__version__'):
            print(f"   ✅ peppi version: {peppi.__version__}")
        
        # Try to access some common peppi functions
        if hasattr(peppi, 'parse'):
            print("   ✅ peppi.parse function available")
        if hasattr(peppi, 'Game'):
            print("   ✅ peppi.Game class available")
            
        return {"success": True, "message": "peppi-py is working!"}
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
    except Exception as e:
        print(f"   ❌ Import error: {e}")
    
    # 6. Check installation logs
    print(f"\n6. Installation Logs:")
    log_files = ['/tmp/peppi_install_method1.log', '/tmp/peppi_install_method2.log', '/tmp/peppi_install_method3.log']
    
    for i, log_file in enumerate(log_files, 1):
        if os.path.exists(log_file):
            print(f"\n   Method {i} Log (last 30 lines):")
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    for line in lines[-30:]:  # Last 30 lines
                        print(f"     {line.rstrip()}")
            except Exception as e:
                print(f"     ❌ Error reading log: {e}")
        else:
            print(f"   Method {i} Log: Not found")
    
    # 7. Manual installation attempt with full output
    print(f"\n7. Manual Installation Attempt:")
    try:
        print("   Attempting fresh installation with proper environment...")
        
        # Set up environment properly
        env = os.environ.copy()
        env['PATH'] = '/root/.cargo/bin:' + env.get('PATH', '')
        env['CARGO_HOME'] = '/root/.cargo'
        env['RUSTUP_HOME'] = '/root/.rustup'
        
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', 
            '--no-build-isolation', '--verbose',
            'git+https://github.com/hohav/peppi-py.git'
        ], capture_output=True, text=True, timeout=600, env=env)
        
        print(f"   Return code: {result.returncode}")
        if result.stdout:
            print("   STDOUT (last 50 lines):")
            stdout_lines = result.stdout.split('\n')
            for line in stdout_lines[-50:]:
                if line.strip():
                    print(f"     {line}")
        
        if result.stderr:
            print("   STDERR (last 30 lines):")
            stderr_lines = result.stderr.split('\n')
            for line in stderr_lines[-30:]:
                if line.strip():
                    print(f"     {line}")
                    
    except subprocess.TimeoutExpired:
        print("   ❌ Installation timed out after 10 minutes")
    except Exception as e:
        print(f"   ❌ Installation error: {e}")
    
    # 8. Final import test
    print(f"\n8. Final Import Test:")
    try:
        import peppi
        print("   ✅ SUCCESS: peppi imported after manual installation!")
        return {"success": True, "message": "peppi-py working after manual install"}
    except Exception as e:
        print(f"   ❌ FAILED: Still cannot import peppi: {e}")
        return {"success": False, "error": str(e)}

@app.local_entrypoint()
def main():
    print("🚀 Starting FIXED peppi-py installation debug")
    
    try:
        result = debug_peppi_comprehensive.remote()
        
        if isinstance(result, dict):
            if result.get('success'):
                print(f"\n🎉 SUCCESS: {result.get('message', 'peppi-py is working')}")
            else:
                print(f"\n💥 FAILED: {result.get('error', 'Unknown error')}")
        else:
            print(f"\n🤔 Debug completed with result: {result}")
            
    except Exception as e:
        print(f"💥 Debug failed: {e}")
    
    print("\n📋 Analysis complete!")
    return True