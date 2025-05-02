#!/bin/bash

# Set script to exit if any command fails
set -e

# Helper script to deploy Jupyter with Mullvad VPN

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if .env file exists
if [ ! -f .env ]; then
    echo "No .env file found. Creating from example..."
    cp .env.example .env
    echo "Please edit .env file with your Mullvad account number and preferred settings"
    exit 1
fi

# Source the .env file
source .env

# Validate Mullvad account number is set
if [ -z "$MULLVAD_ACCOUNT" ] || [ "$MULLVAD_ACCOUNT" = "1234567890123456" ]; then
    echo "Error: Please set a valid MULLVAD_ACCOUNT in the .env file"
    exit 1
fi

# Build and start containers
echo "Building and starting Jupyter + Mullvad VPN container..."
docker-compose up --build -d

# Wait for container to initialize
echo "Waiting for container to initialize..."
sleep 5

# Show logs to get Jupyter URL
echo "Container logs (press Ctrl+C to exit logs):"
docker-compose logs -f jupyter-vpn