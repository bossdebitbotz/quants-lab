#!/bin/bash
# TFT Bot Comprehensive Monitoring Tools Runner
# This script provides an easy way to launch all the bot monitoring tools

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${CYAN}===== TFT Trading Bot Monitoring Suite =====${NC}"
echo "This script provides access to all monitoring tools"
echo

# Function to display menu and get user choice
show_menu() {
    echo -e "${CYAN}Available Monitoring Tools:${NC}"
    echo "1. Bot Monitor (real-time GUI dashboard)"
    echo "2. Position Monitor (tracks position changes)"
    echo "3. PnL Tracker (comprehensive trade analysis)"
    echo "4. Check Position State (shows current position)"
    echo "5. View All Trades (from both database tables)"
    echo "6. Restart Bot Monitor (apply code changes)"
    echo "7. Exit"
    echo
    read -p "Choose a tool (1-7): " choice
    return $choice
}

# Function to run the main bot monitor
run_bot_monitor() {
    echo -e "${YELLOW}Starting Bot Monitor...${NC}"
    cd realtime_tft_bot && tools/run_monitor.sh
}

# Function to run the position monitor
run_position_monitor() {
    echo -e "${YELLOW}Starting Position Monitor...${NC}"
    echo "Press Ctrl+C to exit"
    cd realtime_tft_bot && python tools/monitor_positions.py
}

# Function to run the PnL tracker
run_pnl_tracker() {
    echo -e "${YELLOW}Running PnL Tracker...${NC}"
    cd realtime_tft_bot && python tools/pnl_summary.py
    echo
    echo -e "${CYAN}For more detailed results, run:${NC}"
    echo "python tools/pnl_summary.py --verbose"
    echo
    read -p "Press Enter to continue"
}

# Function to check current position state
check_position_state() {
    echo -e "${YELLOW}Checking Position State...${NC}"
    cd realtime_tft_bot && python tools/check_position_state.py
    echo
    read -p "Press Enter to continue"
}

# Function to view all trades (using the new script)
view_all_trades() {
    echo -e "${YELLOW}Viewing All Trades...${NC}"
    python view_all_trades.py
    echo
    read -p "Press Enter to continue"
}

# Function to restart the bot monitor service
restart_bot_monitor() {
    echo -e "${YELLOW}Restarting Bot Monitor...${NC}"
    
    # Find and kill any running bot monitor processes
    echo "Stopping any running bot monitor processes..."
    pkill -f "python.*bot_monitor.py" || true
    
    echo "Clearing any cached Python files..."
    find realtime_tft_bot/tools -name "*.pyc" -delete
    
    echo -e "${GREEN}Bot monitor has been restarted. Use option 1 to launch it again.${NC}"
    echo
    read -p "Press Enter to continue"
}

# Main loop
while true; do
    clear
    show_menu
    choice=$?
    
    case $choice in
        1)
            run_bot_monitor
            ;;
        2)
            run_position_monitor
            ;;
        3)
            run_pnl_tracker
            ;;
        4)
            check_position_state
            ;;
        5)
            view_all_trades
            ;;
        6)
            restart_bot_monitor
            ;;
        7)
            echo -e "${GREEN}Exiting Monitoring Suite${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid choice. Please try again.${NC}"
            sleep 1
            ;;
    esac
done 