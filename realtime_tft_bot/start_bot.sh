#!/bin/bash
# Script to deploy and start the TFT Bot

# Define color codes for pretty output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}==========================================${NC}"
echo -e "${GREEN}    TFT Bot Deployment Script            ${NC}"
echo -e "${GREEN}==========================================${NC}"

# Check for .env file
if [ ! -f .env ]; then
    echo -e "${YELLOW}No .env file found. Creating from sample...${NC}"
    if [ -f sample.env ]; then
        cp sample.env .env
        echo -e "${GREEN}Created .env file. Please edit it with your configuration.${NC}"
    else
        echo -e "${RED}Error: sample.env file not found!${NC}"
        exit 1
    fi
fi

# Check for models directory
if [ ! -d "../models" ]; then
    echo -e "${YELLOW}Models directory not found. Creating...${NC}"
    mkdir -p ../models
    echo -e "${YELLOW}Warning: You need to place your TFT model files in ../models/${NC}"
fi

# Check Docker and Docker Compose
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed!${NC}"
    echo "Please install Docker and try again"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo -e "${YELLOW}Docker Compose not found, checking for Docker Compose plugin...${NC}"
    if ! docker compose version &> /dev/null; then
        echo -e "${RED}Error: Docker Compose is not installed!${NC}"
        echo "Please install Docker Compose and try again"
        exit 1
    fi
    # Use Docker Compose plugin instead
    COMPOSE_CMD="docker compose"
else
    COMPOSE_CMD="docker-compose"
fi

# Check TimescaleDB connection
echo -e "${GREEN}Checking TimescaleDB connection...${NC}"
source .env

DB_HOST_VALUE=${DB_HOST}
DB_PORT_VALUE=${DB_PORT}
DB_NAME_VALUE=${DB_NAME}
DB_USER_VALUE=${DB_USER}
DB_PASSWORD_VALUE=${DB_PASSWORD}

# Use Docker to run a PostgreSQL client container to test the connection
# This avoids requiring psql to be installed on the host
echo -e "${YELLOW}Testing database connection...${NC}"
docker run --rm --network=host postgres:14 \
    bash -c "PGPASSWORD='${DB_PASSWORD_VALUE}' psql -h ${DB_HOST_VALUE} -p ${DB_PORT_VALUE} -U ${DB_USER_VALUE} -d ${DB_NAME_VALUE} -c 'SELECT version();'" > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Database connection successful!${NC}"
else
    echo -e "${RED}Failed to connect to the database.${NC}"
    echo "Please check your database configuration in .env file"
    echo "Or make sure TimescaleDB is running"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Verify database schema
echo -e "${GREEN}Checking for required database tables...${NC}"
docker run --rm --network=host postgres:14 \
    bash -c "PGPASSWORD='${DB_PASSWORD_VALUE}' psql -h ${DB_HOST_VALUE} -p ${DB_PORT_VALUE} -U ${DB_USER_VALUE} -d ${DB_NAME_VALUE} -c 'SELECT COUNT(*) FROM information_schema.tables WHERE table_name IN (\"order_events\", \"executed_trades\");'" > /dev/null 2>&1

if [ $? -ne 0 ]; then
    echo -e "${YELLOW}Database schema may need to be created.${NC}"
    echo -e "Would you like to apply the database schema now? (y/n) "
    read -p "" -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${GREEN}Applying database schema...${NC}"
        if [ -f "../database_setup/tftbot_schema_fixed.sql" ]; then
            cat ../database_setup/tftbot_schema_fixed.sql | \
            docker run -i --rm --network=host postgres:14 \
                bash -c "PGPASSWORD='${DB_PASSWORD_VALUE}' psql -h ${DB_HOST_VALUE} -p ${DB_PORT_VALUE} -U ${DB_USER_VALUE} -d ${DB_NAME_VALUE}"
            if [ $? -eq 0 ]; then
                echo -e "${GREEN}Database schema applied successfully!${NC}"
            else
                echo -e "${RED}Failed to apply database schema.${NC}"
                exit 1
            fi
        else
            echo -e "${RED}Database schema file not found at ../database_setup/tftbot_schema_fixed.sql${NC}"
            exit 1
        fi
    fi
fi

# Check for model files
DIR_MODEL_PATH=${DIR_MODEL_PATH#/app/}
DIR_MODEL_PATH="../${DIR_MODEL_PATH}"
DOWN_MODEL_PATH=${DOWN_MODEL_PATH#/app/}
DOWN_MODEL_PATH="../${DOWN_MODEL_PATH}"

if [ ! -f "${DIR_MODEL_PATH}" ]; then
    echo -e "${YELLOW}Warning: Directional model file not found at ${DIR_MODEL_PATH}${NC}"
    echo "Please ensure model files are available before starting the bot"
fi

if [ ! -f "${DOWN_MODEL_PATH}" ]; then
    echo -e "${YELLOW}Warning: Downward specialist model file not found at ${DOWN_MODEL_PATH}${NC}"
    echo "Please ensure model files are available before starting the bot"
fi

# Build and start the bot
echo -e "${GREEN}Building and starting the TFT Bot...${NC}"
${COMPOSE_CMD} up --build -d

if [ $? -eq 0 ]; then
    echo -e "${GREEN}TFT Bot started successfully!${NC}"
    echo -e "${GREEN}To view logs: ${YELLOW}docker logs -f realtime_tft_bot${NC}"
    echo -e "${GREEN}To stop the bot: ${YELLOW}docker-compose down${NC}"
else
    echo -e "${RED}Failed to start TFT Bot.${NC}"
    exit 1
fi

echo -e "${GREEN}==========================================${NC}"
echo -e "${GREEN}    Bot is now running!                   ${NC}"
echo -e "${GREEN}==========================================${NC}" 