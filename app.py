# app.py (Main Launcher)

import subprocess
import multiprocessing
import os
import sys
from time import sleep

def run_streamlit():
    """Run the Streamlit frontend."""
    # Use os.path.join for cross-platform compatibility
    frontend_path = os.path.join(os.path.dirname(__file__), "frontend", "app.py")
    # Command to run streamlit. Note: We're not setting the port here,
    # Hugging Face will set the PORT environment variable.
    command = f"streamlit run {frontend_path}"
    subprocess.run(command, shell=True)

def run_flask():
    """Run the Flask backend."""
    backend_path = os.path.join(os.path.dirname(__file__), "backend", "app.py")
    # Command to run flask using gunicorn for production
    command = f"gunicorn --workers 2 --timeout 120 --bind 0.0.0.0:5001 backend.app:app"
    subprocess.run(command, shell=True)

if __name__ == "__main__":
    # Ensure the backend directory is in the Python path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

    # Create and start the processes
    flask_process = multiprocessing.Process(target=run_flask)
    streamlit_process = multiprocessing.Process(target=run_streamlit)
    
    print("Starting Flask backend...")
    flask_process.start()
    
    # Give the backend a moment to start up before launching the frontend
    sleep(15) 
    
    print("Starting Streamlit frontend...")
    streamlit_process.start()
    
    # Wait for processes to complete
    flask_process.join()
    streamlit_process.join()