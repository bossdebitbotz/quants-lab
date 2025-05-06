#!/bin/bash

echo "===== Starting Hummingbot ====="
echo "NOTE: Hummingbot will start with auto-login enabled."
echo "      After Hummingbot starts, run these commands to set up TFT strategy:"
echo ""
echo "  import scripts/register_tft_strategy_v2.py"
echo "  import scripts/register_and_start.py"
echo "  create"
echo "  import_config strategies/tft_strategy_config.yml"
echo "  start"
echo ""
echo "===== Connecting to Hummingbot ====="

docker exec -it hummingbot bash -c "cd /home/hummingbot && conda run -n hummingbot /home/hummingbot/bin/hummingbot.py --auto-login"
