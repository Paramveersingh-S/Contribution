import os
import subprocess

def run_cmd(cmd):
    print(f"==================================================")
    print(f"Running: {cmd}")
    print(f"==================================================")
    subprocess.run(cmd, shell=True, check=True)

if __name__ == "__main__":
    print("TIGRE CUDA Testing Script (Google Colab T4 GPU)")
    
    # Check GPU
    try:
        run_cmd("nvidia-smi")
    except Exception:
        print("Warning: nvidia-smi failed. Ensure you are on a T4 GPU instance.")

    # Clone the repository
    if not os.path.exists("TIGRE"):
        run_cmd("git clone https://github.com/Paramveersingh-S/TIGRE.git")
    
    # Move into the TIGRE directory
    os.chdir("TIGRE")
    
    # Optional: Checkout your specific branch
    # run_cmd("git checkout fix/issue-681")
    
    # Install Python dependencies
    run_cmd("pip install pytest numpy scipy matplotlib")
    
    # Compile and install TIGRE with CUDA support
    # (Colab comes with nvcc pre-installed, so this will build the C++ kernels natively)
    run_cmd("python setup.py install")
    
    # Run the Python Test Suite
    run_cmd("python -m pytest Python/tests/ -v")
