# setup.py
import os
import subprocess
import sys

def install_ffmpeg():
    """Try to install FFmpeg if not available."""
    try:
        # Check if ffmpeg exists
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        print("FFmpeg is already installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("FFmpeg not found, attempting to install...")
        # Try to install from a custom source
        # Note: This may not work on Streamlit Cloud due to restrictions
        os.system('apt-get update && apt-get install -y ffmpeg')

if __name__ == "__main__":
    install_ffmpeg()