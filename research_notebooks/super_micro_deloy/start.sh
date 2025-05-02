#!/bin/bash
set -e

# Log function for debugging
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

log "Starting container..."

# Check if Mullvad account number is provided
if [ -z "$MULLVAD_ACCOUNT" ]; then
    log "Error: MULLVAD_ACCOUNT environment variable is not set"
    exit 1
fi

# Check if location is provided, default to Singapore if not
MULLVAD_LOCATION=${MULLVAD_LOCATION:-"sg"}
log "Using VPN location: $MULLVAD_LOCATION"

# Log in to Mullvad
log "Logging in to Mullvad..."
mullvad account login "$MULLVAD_ACCOUNT"

# Set relay location based on the selected Asian location
log "Setting relay location..."
mullvad relay set location "$MULLVAD_LOCATION"

# Connect to VPN
log "Connecting to Mullvad VPN..."
mullvad connect

# Wait until VPN is connected
log "Waiting for VPN connection..."
while ! mullvad status | grep -q "Connected"; do
    log "Waiting for VPN connection..."
    sleep 2
done

log "VPN connected successfully"

# Check if jupyter password is provided
if [ -n "$JUPYTER_PASSWORD" ]; then
    log "Setting up Jupyter with password authentication"
    # Generate Jupyter config if it doesn't exist
    if [ ! -f ~/.jupyter/jupyter_notebook_config.py ]; then
        jupyter notebook --generate-config
        # Set password
        python -c "from notebook.auth import passwd; print(passwd('$JUPYTER_PASSWORD'))" > /tmp/jupyter_pass
        HASHED_PASSWORD=$(cat /tmp/jupyter_pass)
        echo "c.NotebookApp.password = '$HASHED_PASSWORD'" >> ~/.jupyter/jupyter_notebook_config.py
        rm /tmp/jupyter_pass
    fi
else
    log "No password set, Jupyter will use token authentication"
fi

# Start Jupyter Notebook
log "Starting Jupyter Notebook..."
cd /quants-lab
conda run -n quants-lab jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root 