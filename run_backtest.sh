#!/bin/bash

# =====================================================================
# Run TFT Ensemble Backtest on 7 Days of Orderbook Data
# =====================================================================

# Create timestamp for output directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="backtest_results/tft_ensemble_${TIMESTAMP}"

# Default models
DIRECTIONAL_MODEL="models/directional_tft_20250502_134913.pt"
DOWNWARD_MODEL="models/downward_specialist_tft_20250502_140300.pt"

# Default trading parameters
THRESHOLD=0.002
STOP_LOSS=0.002
TAKE_PROFIT=0.004
MAX_HOLDING=8
FEE=0.0001  # 0.01%

# Ensure scripts are executable
chmod +x backtest_tft_on_orderbook.py

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=== Starting TFT Ensemble Backtest ==="
echo "Directional Model: $DIRECTIONAL_MODEL"
echo "Downward Model: $DOWNWARD_MODEL"
echo "Output Directory: $OUTPUT_DIR"
echo "Started at: $(date)"
echo

# Run the backtest
python backtest_tft_on_orderbook.py \
  --days 7 \
  --directional-model "$DIRECTIONAL_MODEL" \
  --downward-model "$DOWNWARD_MODEL" \
  --ensemble-method "selective" \
  --threshold "$THRESHOLD" \
  --stop-loss "$STOP_LOSS" \
  --take-profit "$TAKE_PROFIT" \
  --max-holding "$MAX_HOLDING" \
  --fee "$FEE" \
  --context-length 30 \
  --output-dir "$OUTPUT_DIR" \
  --position-sizing \
  --dynamic-threshold

# Check if backtest was successful
if [ $? -eq 0 ]; then
  echo
  echo "=== Backtest Completed Successfully ==="
  echo "Results saved to: $OUTPUT_DIR"
  echo "Completed at: $(date)"
  
  # Print summary from the results file if available
  if [ -f "$OUTPUT_DIR/backtest_results.json" ]; then
    echo
    echo "=== Backtest Summary ==="
    # Extract key metrics with jq if available, otherwise cat the file
    if command -v jq &> /dev/null; then
      echo "Total Trades: $(jq '.total_trades' "$OUTPUT_DIR/backtest_results.json")"
      echo "Win Rate: $(jq '.win_rate' "$OUTPUT_DIR/backtest_results.json" | awk '{print $1*100 "%"}')"
      echo "Total Return: $(jq '.total_return' "$OUTPUT_DIR/backtest_results.json" | awk '{print $1*100 "%"}')"
      echo "Sharpe Ratio: $(jq '.sharpe_ratio' "$OUTPUT_DIR/backtest_results.json")"
      echo "Final Equity: $(jq '.final_equity' "$OUTPUT_DIR/backtest_results.json")"
    else
      cat "$OUTPUT_DIR/backtest_results.json"
    fi
  fi
else
  echo
  echo "=== Backtest Failed ==="
  echo "Check logs for errors"
  echo "Failed at: $(date)"
fi 