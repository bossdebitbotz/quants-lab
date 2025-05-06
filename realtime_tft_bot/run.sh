#!/bin/bash
# Script to run the real-time TFT bot

# Ensure virtual environment is activated if using one
# source venv/bin/activate

# Verify the database connection
echo "Verifying database connection..."
python -c "
import psycopg2
from psycopg2 import sql
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get database configuration
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = int(os.getenv('DB_PORT', 5441))
DB_NAME = os.getenv('DB_NAME', 'backtest_db')
DB_USER = os.getenv('DB_USER', 'backtest_user')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'backtest_password')

try:
    # Connect to the database
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    
    # Test query
    with conn.cursor() as cursor:
        cursor.execute('SELECT current_timestamp')
        timestamp = cursor.fetchone()[0]
        print(f'Database connection successful! Server time: {timestamp}')
    
    # Close connection
    conn.close()
    
except Exception as e:
    print(f'Database connection failed: {str(e)}')
    exit(1)
"

# Check if verification was successful
if [ $? -ne 0 ]; then
    echo "Database verification failed. Please check your configuration."
    exit 1
fi

# Run the bot
echo "Starting TFT Bot..."
python realtime_tft_bot.py "$@" 