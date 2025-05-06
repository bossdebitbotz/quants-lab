import os
import psycopg2
from psycopg2 import pool
from dotenv import load_dotenv
import logging
import pandas as pd
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DB_CONNECTION_POOL = None

def initialize_db_pool():
    """Initializes the database connection pool."""
    global DB_CONNECTION_POOL
    if DB_CONNECTION_POOL:
        return

    try:
        # Load environment variables from realtime_tft_bot/.env
        dotenv_path = os.path.join(os.path.dirname(__file__), 'realtime_tft_bot', '.env')
        if os.path.exists(dotenv_path):
            load_dotenv(dotenv_path=dotenv_path)
            logging.info(f"Loaded environment variables from: {dotenv_path}")
        else:
            logging.warning(f".env file not found at {dotenv_path}. Relying on existing environment variables.")

        # Use localhost and port 5441 by default for host connection
        db_name = os.getenv("DB_NAME")
        db_user = os.getenv("DB_USER")
        db_password = os.getenv("DB_PASSWORD")
        db_host = os.getenv("DB_HOST_FROM_HOST", "localhost") # Host connection default
        db_port = os.getenv("DB_PORT_FROM_HOST", "5441") # Host connection default

        if not all([db_name, db_user, db_password, db_host, db_port]):
            logging.error("Database connection details missing in environment variables (DB_NAME, DB_USER, DB_PASSWORD required; DB_HOST_FROM_HOST defaults to localhost, DB_PORT_FROM_HOST defaults to 5441).")
            return False

        logging.info(f"Attempting to connect to database: host={db_host}, port={db_port}, dbname={db_name}")

        DB_CONNECTION_POOL = psycopg2.pool.SimpleConnectionPool(
            minconn=1,
            maxconn=5,
            host=db_host,
            port=db_port,
            database=db_name,
            user=db_user,
            password=db_password
        )
        # Test connection
        conn = DB_CONNECTION_POOL.getconn()
        logging.info("Database connection pool initialized successfully.")
        DB_CONNECTION_POOL.putconn(conn)
        return True

    except psycopg2.OperationalError as e:
        logging.error(f"Database connection failed: {e}")
        logging.error("Ensure the database container is running and the port (default 5441) is mapped correctly to the host.")
        DB_CONNECTION_POOL = None
        return False
    except Exception as e:
        logging.error(f"Failed to initialize database pool: {e}")
        DB_CONNECTION_POOL = None
        return False

def close_db_pool():
    """Closes all connections in the pool."""
    global DB_CONNECTION_POOL
    if DB_CONNECTION_POOL:
        DB_CONNECTION_POOL.closeall()
        logging.info("Database connection pool closed.")
        DB_CONNECTION_POOL = None

def run_query(query, params=None, fetch="many", fetch_count=1000):
    """Executes a query and returns the results."""
    if not DB_CONNECTION_POOL:
        logging.error("Database pool not initialized.")
        return None

    conn = None
    try:
        conn = DB_CONNECTION_POOL.getconn()
        with conn.cursor() as cur:
            cur.execute(query, params)
            if fetch == "one":
                return cur.fetchone()
            elif fetch == "all":
                return cur.fetchall()
            elif fetch == "many":
                return cur.fetchmany(fetch_count)
            elif fetch == "none": # For INSERT/UPDATE/DELETE without RETURNING
                conn.commit()
                return None
            else:
                return None # Default case
    except psycopg2.Error as e:
        logging.error(f"Database query failed: {e}")
        if conn:
            conn.rollback() # Rollback on error
        return None
    finally:
        if conn:
            DB_CONNECTION_POOL.putconn(conn)

def get_prediction_stats():
    """Queries statistics about TFT predictions."""
    stats = {}
    # Count
    result = run_query("SELECT COUNT(*) FROM tft_predictions;", fetch="one")
    stats['count'] = result[0] if result else 0

    if stats['count'] > 0:
        # Time Range
        result = run_query("SELECT MIN(prediction_timestamp), MAX(prediction_timestamp) FROM tft_predictions;", fetch="one")
        if result:
            stats['min_time'] = result[0]
            stats['max_time'] = result[1]

        # Prediction Value Stats
        result = run_query("SELECT ensemble_prediction FROM tft_predictions ORDER BY prediction_timestamp DESC LIMIT 10000;", fetch="all") # Sample last 10k
        if result:
            # Convert Decimal to float before creating Series
            float_preds = [float(r[0]) for r in result if r[0] is not None]
            if not float_preds:
                 return stats # Return early if no valid numeric predictions
                 
            preds = pd.Series(float_preds)
            stats['prediction_mean'] = preds.mean()
            stats['prediction_std'] = preds.std()
            stats['prediction_min'] = preds.min()
            stats['prediction_max'] = preds.max()
            stats['prediction_median'] = preds.median()

    return stats

def get_decision_stats():
    """Queries statistics about trade decisions."""
    stats = {}
    # Counts per decision type
    result = run_query("SELECT decision, COUNT(*) FROM trade_decisions GROUP BY decision;", fetch="all")
    stats['counts_by_type'] = dict(result) if result else {}
    stats['total_decisions'] = sum(stats['counts_by_type'].values())

    if stats['total_decisions'] > 0:
        # Time Range
        result = run_query("SELECT MIN(decision_timestamp), MAX(decision_timestamp) FROM trade_decisions;", fetch="one")
        if result:
            stats['min_time'] = result[0]
            stats['max_time'] = result[1]

    return stats

def get_executed_trade_stats():
    """Queries statistics about executed trades."""
    stats = {}
    # Count
    result = run_query("SELECT COUNT(*) FROM executed_trades;", fetch="one")
    stats['count'] = result[0] if result else 0

    if stats['count'] > 0:
        # Time Range
        result = run_query("SELECT MIN(execution_timestamp), MAX(execution_timestamp) FROM executed_trades;", fetch="one")
        if result:
            stats['min_time'] = result[0]
            stats['max_time'] = result[1]
        # Basic PnL stats if available
        result = run_query("SELECT pnl FROM executed_trades WHERE action = 'CLOSE';", fetch="all")
        if result:
             pnls = pd.Series([r[0] for r in result if r[0] is not None])
             if not pnls.empty:
                 stats['closed_trade_count'] = len(pnls)
                 stats['avg_pnl_per_trade'] = pnls.mean()
                 stats['total_pnl_trades'] = pnls.sum()

    return stats

def get_position_state():
    """Queries the current position state."""
    # Assuming only one trading pair for now
    trading_pair = os.getenv("TRADING_PAIR", "WLD-USDT") # Get from env or default
    result = run_query("SELECT position, entry_price, position_size, entry_timestamp, holding_periods FROM positions WHERE trading_pair = %s;", params=(trading_pair,), fetch="one")
    if result:
        return {
            'position': result[0],
            'entry_price': result[1],
            'position_size': result[2],
            'entry_timestamp': result[3],
            'holding_periods': result[4]
        }
    else:
        return {'position': 'UNKNOWN', 'reason': f'No position found for {trading_pair}'}

def get_performance_metrics():
    """Queries the latest performance metrics."""
    result = run_query("SELECT timestamp, cumulative_pnl, total_trades, winning_trades, max_drawdown FROM performance_metrics ORDER BY timestamp DESC LIMIT 1;", fetch="one")
    if result:
        return {
            'timestamp': result[0],
            'cumulative_pnl': result[1],
            'total_trades': result[2],
            'winning_trades': result[3],
            'max_drawdown': result[4]
        }
    else:
        return {'cumulative_pnl': 0.0, 'reason': 'No performance metrics found'}

def format_timedelta(start_time, end_time):
    if start_time and end_time:
        delta = end_time - start_time
        return str(delta)
    return "N/A"

def format_stat(value, precision=6):
    """Formats a statistic, handling non-numeric values gracefully."""
    if isinstance(value, (int, float)):
        return f"{value:.{precision}f}"
    return "N/A"

if __name__ == "__main__":
    if not initialize_db_pool():
        print("Exiting due to database connection failure.")
        exit(1)

    print("\n--- Database Statistics ---")

    # Predictions
    print("\n[Predictions (tft_predictions)]")
    pred_stats = get_prediction_stats()
    print(f"  Total Predictions: {pred_stats.get('count', 'N/A')}")
    if pred_stats.get('count', 0) > 0:
        print(f"  Time Range: {pred_stats.get('min_time')} -> {pred_stats.get('max_time')} ({format_timedelta(pred_stats.get('min_time'), pred_stats.get('max_time'))})")
        print(f"  Ensemble Prediction Stats (last 10k):")
        print(f"    Mean:   {format_stat(pred_stats.get('prediction_mean'))}")
        print(f"    Std Dev:{format_stat(pred_stats.get('prediction_std'))}")
        print(f"    Min:    {format_stat(pred_stats.get('prediction_min'))}")
        print(f"    Max:    {format_stat(pred_stats.get('prediction_max'))}")
        print(f"    Median: {format_stat(pred_stats.get('prediction_median'))}")

    # Decisions
    print("\n[Decisions (trade_decisions)]")
    dec_stats = get_decision_stats()
    print(f"  Total Decisions: {dec_stats.get('total_decisions', 'N/A')}")
    if dec_stats.get('total_decisions', 0) > 0:
         print(f"  Time Range: {dec_stats.get('min_time')} -> {dec_stats.get('max_time')} ({format_timedelta(dec_stats.get('min_time'), dec_stats.get('max_time'))})")
         print(f"  Counts by Type:")
         for dtype, count in dec_stats.get('counts_by_type', {}).items():
             print(f"    {dtype}: {count}")

    # Executed Trades
    print("\n[Executed Trades (executed_trades)]")
    trade_stats = get_executed_trade_stats()
    print(f"  Total Executed Orders Logged: {trade_stats.get('count', 'N/A')}")
    if trade_stats.get('count', 0) > 0:
        print(f"  Time Range: {trade_stats.get('min_time')} -> {trade_stats.get('max_time')} ({format_timedelta(trade_stats.get('min_time'), trade_stats.get('max_time'))})")
        print(f"  Closed Trades Logged: {trade_stats.get('closed_trade_count', 0)}")
        print(f"  Total PnL (from trades table): {format_stat(trade_stats.get('total_pnl_trades'))}")
        print(f"  Avg PnL (from trades table):   {format_stat(trade_stats.get('avg_pnl_per_trade'))}")


    # Position State
    print("\n[Current Position (positions)]")
    pos_state = get_position_state()
    print(f"  Trading Pair: {os.getenv('TRADING_PAIR', 'WLD-USDT')}")
    print(f"  Position: {pos_state.get('position', 'N/A')}")
    if pos_state.get('position') not in ['NONE', 'UNKNOWN']:
        print(f"  Entry Price: {pos_state.get('entry_price', 'N/A')}")
        print(f"  Position Size: {pos_state.get('position_size', 'N/A')}")
        print(f"  Entry Timestamp: {pos_state.get('entry_timestamp', 'N/A')}")
        print(f"  Holding Periods: {pos_state.get('holding_periods', 'N/A')}")
    elif pos_state.get('reason'):
         print(f"  Details: {pos_state.get('reason')}")


    # Performance Metrics
    print("\n[Performance Metrics (performance_metrics) - Latest Record]")
    perf_metrics = get_performance_metrics()
    if 'reason' in perf_metrics:
        print(f"  Status: {perf_metrics['reason']}")
    else:
        print(f"  Timestamp: {perf_metrics.get('timestamp')}")
        print(f"  Cumulative PnL: {format_stat(perf_metrics.get('cumulative_pnl'))}")
        print(f"  Total Trades: {perf_metrics.get('total_trades', 'N/A')}")
        print(f"  Winning Trades: {perf_metrics.get('winning_trades', 'N/A')}")
        print(f"  Max Drawdown: {format_stat(perf_metrics.get('max_drawdown'))}")

    print("\n--- End Statistics ---")

    close_db_pool() 