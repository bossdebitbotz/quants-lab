#!/usr/bin/env python3
import os
import sys
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("bot_starter")

# Hard-code the configuration overrides
os.environ['DIR_MODEL_PATH'] = "/Volumes/Elite_Frosty /quants-lab/models/directional_tft_20250502_134913.pt"
os.environ['DOWN_MODEL_PATH'] = "/Volumes/Elite_Frosty /quants-lab/models/downward_specialist_tft_20250502_140300.pt"
os.environ['DB_HOST'] = "localhost"
os.environ['DEPTH_STREAM_INTERVAL'] = "1000ms"  # Use 1000ms instead of 100ms!
os.environ['TORCH_USE_RTLD_GLOBAL'] = "TRUE"

# Load PyTorch and add security exceptions
logger.info("Configuring PyTorch...")
try:
    import torch
    torch.serialization.add_safe_globals(["lightning_fabric.utilities.data.AttributeDict"])
    logger.info("PyTorch loaded successfully")
except ImportError:
    logger.error("Failed to import PyTorch. Please install with: conda install pytorch -c pytorch")
    sys.exit(1)

# Wait to ensure rate limits reset
logger.info("Waiting 5 minutes for Binance rate limits to reset...")
time.sleep(300)

# Launch the bot
logger.info("Starting TFT bot...")
cmd = f"python {os.path.join(os.path.dirname(__file__), 'realtime_tft_bot', 'realtime_tft_bot.py')}"
sys.exit(os.system(cmd))
