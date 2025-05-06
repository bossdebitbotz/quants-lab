#!/bin/bash
# Setup script for TFT Bot Monitor
# This script installs the required dependencies for the monitoring tool

echo "Setting up TFT Bot Monitoring Tool..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but could not be found"
    exit 1
fi

# Install required Python packages
echo "Installing required Python packages..."
python3 -m pip install pandas tabulate psutil psycopg2-binary python-dotenv pyyaml numpy torch

# Create a symbolic link to the parent config file if it doesn't exist
if [ ! -f "./config.py" ]; then
    echo "Creating symbolic link to config.py in parent directory..."
    ln -sf ../config.py ./config.py
fi

echo "Setup complete! You can now run the monitor using ./run_monitor.sh" 