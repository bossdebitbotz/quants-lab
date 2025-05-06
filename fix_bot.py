#!/usr/bin/env python3
import os
import sys
import subprocess
import time
import json

# Force override paths in config system
CONFIG_FILE = "realtime_tft_bot/config.py"

# Create a patch to modify the config loading
with open(CONFIG_FILE, 'r') as f:
    config_content = f.read()

patched_content = config_content.replace(
    'config["DB_HOST"] = os.environ.get("DB_HOST", config.get("DB_HOST", "host.docker.internal"))',
    'config["DB_HOST"] = "localhost"'
)
patched_content = patched_content.replace(
    'config["DIR_MODEL_PATH"] = os.environ.get("DIR_MODEL_PATH", config.get("DIR_MODEL_PATH"))',
    'config["DIR_MODEL_PATH"] = "/Volumes/Elite_Frosty /quants-lab/models/directional_tft_20250502_134913.pt"'
)
patched_content = patched_content.replace(
    'config["DOWN_MODEL_PATH"] = os.environ.get("DOWN_MODEL_PATH", config.get("DOWN_MODEL_PATH"))',
    'config["DOWN_MODEL_PATH"] = "/Volumes/Elite_Frosty /quants-lab/models/downward_specialist_tft_20250502_140300.pt"'
)
patched_content = patched_content.replace(
    'config["DEPTH_STREAM_INTERVAL"] = os.environ.get("DEPTH_STREAM_INTERVAL", config.get("DEPTH_STREAM_INTERVAL", "100ms"))',
    'config["DEPTH_STREAM_INTERVAL"] = "1000ms"'
)

with open(CONFIG_FILE, 'w') as f:
    f.write(patched_content)

print("Configuration patched to use local paths and proper settings")

# Fix PyTorch security settings
print("Initializing PyTorch security settings...")
with open("fix_torch.py", "w") as f:
    f.write('import torch; torch.serialization.add_safe_globals(["lightning_fabric.utilities.data.AttributeDict"])')

subprocess.run([sys.executable, "fix_torch.py"])

# Wait for rate limits to reset
print("Waiting 5 minutes for Binance rate limits to reset...")
time.sleep(300)

# Start the bot
print("\nStarting TFT trading bot with fixed configuration...")
cmd = f"{sys.executable} realtime_tft_bot/realtime_tft_bot.py"
subprocess.run(cmd, shell=True)
