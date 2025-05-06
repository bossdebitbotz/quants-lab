#!/usr/bin/env python3
import os
import sys
import logging
import time
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bot_starter")

# Add PyTorch security exceptions
torch.serialization.add_safe_globals(["lightning_fabric.utilities.data.AttributeDict"])

# Set correct local paths
model_dir = "/Volumes/Elite_Frosty /quants-lab/models"
dir_model = f"{model_dir}/directional_tft_20250502_134913.pt"
down_model = f"{model_dir}/downward_specialist_tft_20250502_140300.pt"

# Set environment variables
os.environ['DIR_MODEL_PATH'] = dir_model
os.environ['DOWN_MODEL_PATH'] = down_model
os.environ['DB_HOST'] = 'localhost'
os.environ['DB_PORT'] = '5441'
os.environ['DEPTH_STREAM_INTERVAL'] = '1000ms'  # Use slower updates to avoid rate limits
os.environ['TORCH_USE_RTLD_GLOBAL'] = 'TRUE'

# Wait for rate limits to reset
logger.info("Waiting 5 minutes for Binance rate limits to reset...")
time.sleep(300)

# Launch the bot
logger.info("Starting TFT bot with fixed configuration...")
os.system("python realtime_tft_bot/realtime_tft_bot.py")
