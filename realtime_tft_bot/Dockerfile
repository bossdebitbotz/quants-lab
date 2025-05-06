FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY *.py ./
COPY utils/ ./utils/

# Copy models directory (assuming models will be mounted)
RUN mkdir -p /app/models

# Copy model definition file
COPY temporal_fusion_transformer.py /app/models/

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the bot
CMD ["python", "realtime_tft_bot.py"] 