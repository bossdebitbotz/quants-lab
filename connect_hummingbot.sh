#!/bin/bash

# Check if hummingbot container is running
if [ "$(docker ps -q -f name=hummingbot)" ]; then
    echo "Connecting to Hummingbot..."
    echo "NOTE: After connecting, run 'import scripts/register_tft_strategy.py' to register the TFT strategy"
    echo "Then create and start the strategy with 'create tft_strategy' and 'start'"
    echo "----------------------------------------------------------------------------------------"
    docker exec -it hummingbot bash -c "cd /home/hummingbot && conda run -n hummingbot /home/hummingbot/bin/hummingbot.py"
else
    echo "Hummingbot container is not running. Start it with 'docker start hummingbot' first."
fi
