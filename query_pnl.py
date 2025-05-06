import os
import psycopg2
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_latest_pnl():
    """Connects to the database and retrieves the latest cumulative PnL."""
    try:
        # Load environment variables from .env file in the parent directory
        dotenv_path = os.path.join(os.path.dirname(__file__), '..', 'realtime_tft_bot', '.env')
        # Fallback to current directory if not found
        if not os.path.exists(dotenv_path):
             dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
        
        if os.path.exists(dotenv_path):
            load_dotenv(dotenv_path=dotenv_path)
            logging.info(f"Loaded environment variables from: {dotenv_path}")
        else:
            logging.warning(".env file not found in expected locations.")
            # Attempt to use environment variables directly if .env is missing

        # Construct the DSN (Data Source Name)
        db_name = os.getenv("DB_NAME")
        db_user = os.getenv("DB_USER")
        db_password = os.getenv("DB_PASSWORD")
        db_host = os.getenv("DB_HOST", "db") # Default to 'db' for Docker service name
        db_port = os.getenv("DB_PORT", "5432")

        if not all([db_name, db_user, db_password, db_host, db_port]):
            logging.error("Database connection details missing in environment variables.")
            return None

        dsn = f"dbname='{db_name}' user='{db_user}' password='{db_password}' host='{db_host}' port='{db_port}'"
        logging.info(f"Attempting to connect to database: host={db_host}, port={db_port}, dbname={db_name}")

        conn = None
        try:
            conn = psycopg2.connect(dsn)
            logging.info("Database connection successful.")
            cur = conn.cursor()

            # Query the latest cumulative PnL
            query = "SELECT cumulative_pnl FROM performance_metrics ORDER BY timestamp DESC LIMIT 1;"
            logging.info(f"Executing query: {query}")
            cur.execute(query)
            result = cur.fetchone()

            cur.close()

            if result:
                latest_pnl = result[0]
                logging.info(f"Latest cumulative PnL: {latest_pnl}")
                return latest_pnl
            else:
                logging.info("No PnL data found in performance_metrics table.")
                return 0.0 # Return 0 if no records exist yet

        except psycopg2.Error as e:
            logging.error(f"Database query failed: {e}")
            return None
        finally:
            if conn:
                conn.close()
                logging.info("Database connection closed.")

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return None

if __name__ == "__main__":
    pnl = get_latest_pnl()
    if pnl is not None:
        print(f"Latest Cumulative PnL: {pnl}")
    else:
        print("Failed to retrieve PnL.") 