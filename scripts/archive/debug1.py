import modal
import subprocess
import sys

# Create Modal app
app = modal.App("peppi-debug-fixed")

# Define image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("curl", "build-essential", "pkg-config", "libssl-dev", "git")
    .run_commands(
        # Install Rust
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
        "echo 'source ~/.cargo/env' >> ~/.bashrc"
    )
    .env({"PATH": "/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"})
    .pip_install("maturin")
    # Try to install peppi-py 0.6.0 in the image
    .run_commands("pip install peppi-py==0.6.0 || echo 'Will try in function'")
)

@app.function(image=image, timeout=600)
def debug_peppi_fixed():
    """Fixed debug function with correct import syntax"""
    
    def test_import():
        """Test the CORRECT import syntax"""
        try:
            # FIXED: Use peppi_py not peppi
            from peppi_py import read_slippi, read_peppi
            print("✅ SUCCESS: peppi_py imported correctly!")
            
            # Test functions
            assert callable(read_slippi), "read_slippi not callable"
            assert callable(read_peppi), "read_peppi not callable"
            print("✅ SUCCESS: peppi_py functions are available")
            
            # Try to get version
            try:
                import peppi_py
                version = getattr(peppi_py, '__version__', 'version unknown')
                print(f"✅ peppi_py version: {version}")
            except:
                print("✅ peppi_py working (version info unavailable)")
            
            return True
            
        except ImportError as e:
            print(f"❌ Import failed: {e}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return False
    
    print("🚀 Starting FIXED peppi-py 0.6.0 installation debug")
    print("🔍 === COMPREHENSIVE PEPPI-PY DEBUG (FIXED) ===\n")
    
    # Check environment
    import os
    print("1. Environment Check:")
    print(f"   Python: {sys.version}")
    print(f"   PATH: {os.environ.get('PATH', 'Not set')}")
    
    # Check if Rust is available
    try:
        result = subprocess.run(['rustc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"   ✅ Rust: {result.stdout.strip()}")
        else:
            print("   ❌ Rust not found")
    except:
        print("   ❌ Rust not available")
    
    # Check current packages
    print("\n2. Current Package Check:")
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', 'list'], capture_output=True, text=True)
        if 'peppi-py' in result.stdout:
            peppi_lines = [line for line in result.stdout.split('\n') if 'peppi' in line.lower()]
            print("   ✅ Found peppi packages:")
            for line in peppi_lines:
                print(f"     {line}")
        else:
            print("   ❌ No peppi packages found")
    except Exception as e:
        print(f"   ❌ Could not check packages: {e}")
    
    # Initial import test
    print("\n3. Initial Import Test:")
    if test_import():
        print("✅ SUCCESS: peppi_py already working!")
        return {"status": "success", "message": "peppi_py already installed and working"}
    
    # Try installation methods
    print("\n4. Installation Attempts:")
    
    installation_methods = [
        {
            "name": "Direct install peppi-py==0.6.0",
            "cmd": [sys.executable, "-m", "pip", "install", "peppi-py==0.6.0", "--force-reinstall"]
        },
        {
            "name": "Install with maturin first",
            "cmd": [sys.executable, "-m", "pip", "install", "maturin", "peppi-py==0.6.0", "--force-reinstall"]
        },
        {
            "name": "From git with version tag",
            "cmd": [sys.executable, "-m", "pip", "install", "git+https://github.com/hohav/peppi-py.git@v0.6.0", "--force-reinstall"]
        }
    ]
    
    for i, method in enumerate(installation_methods, 1):
        print(f"\n   Method {i}: {method['name']}")
        
        try:
            # Run installation
            result = subprocess.run(method['cmd'], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print(f"   ✅ Installation completed")
                
                # Test import immediately
                if test_import():
                    print(f"   ✅ SUCCESS: peppi_py working after {method['name']}!")
                    return {
                        "status": "success", 
                        "method": method['name'],
                        "message": "peppi_py successfully installed and imported"
                    }
                else:
                    print(f"   ❌ Import still failed after installation")
            else:
                print(f"   ❌ Installation failed")
                if result.stderr:
                    error_lines = result.stderr.strip().split('\n')[-5:]  # Last 5 lines
                    for line in error_lines:
                        print(f"     ERROR: {line}")
                        
        except subprocess.TimeoutExpired:
            print(f"   ❌ Installation timed out")
        except Exception as e:
            print(f"   ❌ Installation error: {e}")
    
    # Final attempt - manual installation
    print("\n5. Final Manual Attempt:")
    try:
        # Ensure maturin is installed
        subprocess.run([sys.executable, "-m", "pip", "install", "maturin"], check=False)
        
        # Try building from source
        print("   Attempting to build from source...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "git+https://github.com/hohav/peppi-py.git", 
            "--no-binary=peppi-py",
            "--force-reinstall"
        ], capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            print("   ✅ Source build completed")
            if test_import():
                print("   ✅ SUCCESS: peppi_py working after source build!")
                return {
                    "status": "success",
                    "method": "source build",
                    "message": "peppi_py built from source and working"
                }
        else:
            print("   ❌ Source build failed")
            if result.stderr:
                print("   Last few error lines:")
                for line in result.stderr.strip().split('\n')[-3:]:
                    print(f"     {line}")
                    
    except Exception as e:
        print(f"   ❌ Manual build error: {e}")
    
    # Final import test
    print("\n6. Final Import Test:")
    if test_import():
        print("✅ MIRACULOUS SUCCESS: peppi_py is now working!")
        return {"status": "success", "message": "peppi_py working (unclear which method succeeded)"}
    
    print("\n💥 FAILED: All methods failed")
    print("\n🔧 DEBUGGING INFO:")
    print("   - Package installs as 'peppi-py' but imports as 'peppi_py'")
    print("   - Make sure to use: from peppi_py import read_slippi, read_peppi")
    print("   - Version 0.6.0 specifically requested")
    
    return {"status": "failed", "message": "All installation methods failed"}

# Local function to run the debug
@app.local_entrypoint()
def main():
    """Run the debug function"""
    result = debug_peppi_fixed.remote()
    print(f"\n🎯 Final Result: {result}")
    
    if result.get("status") == "success":
        print("✅ peppi-py 0.6.0 is now working!")
        print("   Use: from peppi_py import read_slippi, read_peppi")
    else:
        print("❌ Installation failed - see debug output above")

if __name__ == "__main__":
    main()