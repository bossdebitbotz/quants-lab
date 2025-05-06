#!/bin/bash
# TFT Bot Monitor Runner
# This script provides an easy way to launch the bot monitor tool

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but could not be found"
    exit 1
fi

# Clean up __pycache__ directories for a fresh start
echo "Cleaning Python cache..."
find "$SCRIPT_DIR/.." -name "__pycache__" -type d -exec rm -rf {} +
find "$SCRIPT_DIR" -name "*.pyc" -type f -delete

# Check for required packages
echo "Checking dependencies..."
if ! python3 -c "import pandas" 2>/dev/null; then
    echo "Error: pandas package is not installed"
    echo "Please run './setup_monitor.sh' first to install required dependencies"
    exit 1
fi

if ! python3 -c "import tabulate" 2>/dev/null; then
    echo "Error: tabulate package is not installed"
    echo "Please run './setup_monitor.sh' first to install required dependencies"
    exit 1
fi

if ! python3 -c "import psycopg2" 2>/dev/null; then
    echo "Error: psycopg2 package is not installed"
    echo "Please run './setup_monitor.sh' first to install required dependencies"
    exit 1
fi

if ! python3 -c "import dotenv" 2>/dev/null; then
    echo "Error: python-dotenv package is not installed"
    echo "Please run './setup_monitor.sh' first to install required dependencies"
    exit 1
fi

# Check if config.py exists in the current directory
if [ ! -f "./config.py" ]; then
    echo "Error: config.py not found in the tools directory"
    echo "Running setup_monitor.sh to create symbolic link..."
    ./setup_monitor.sh
fi

# Default refresh interval
REFRESH=5

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--refresh)
            REFRESH="$2"
            shift 2
            ;;
        -h|--help)
            echo "TFT Bot Monitor"
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  -r, --refresh SECONDS   Set refresh interval in seconds (default: 5)"
            echo "  -h, --help              Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help to see available options"
            exit 1
            ;;
    esac
done

# Print header
echo "Starting TFT Bot Monitor with refresh interval of ${REFRESH}s"
echo "Press Ctrl+C to exit"
echo

# Kill any running instances
pkill -f "python.*bot_monitor.py" || true
sleep 1

# Run the monitor script
python3 "${SCRIPT_DIR}/bot_monitor.py" --refresh "$REFRESH" 