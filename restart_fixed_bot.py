#!/usr/bin/env python3
"""
Restart the TFT bot with the fixed feature calculator code
"""
import os
import sys
import subprocess
import time
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bot_restart.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("BotRestart")

def run_command(command):
    """Run a shell command and log the output"""
    logger.info(f"Running command: {command}")
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        logger.info(f"Command succeeded with output:\n{result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}:\n{e.stderr}")
        return False

def stop_running_bot():
    """Kill any running bot processes"""
    logger.info("Stopping any running bot processes...")
    run_command("pkill -f 'python.*realtime_tft_bot.py'")
    # Give time for processes to terminate
    time.sleep(3)

def start_bot():
    """Start the bot in the background"""
    logger.info("Starting the TFT bot...")
    start_cmd = "cd realtime_tft_bot && python realtime_tft_bot.py > run_output.log 2>&1 &"
    os.system(start_cmd)
    logger.info("Bot startup command issued")
    return True

def restart_bot():
    """Restart the bot"""
    logger.info("=== Restarting TFT Bot with Fixed Feature Calculator ===")
    
    # Stop any running bot processes
    stop_running_bot()
    
    # Start the bot
    if start_bot():
        logger.info("Bot started successfully!")
        return True
    else:
        logger.error("Failed to start bot!")
        return False

if __name__ == "__main__":
    restart_bot() 