FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for the model server
RUN pip install --no-cache-dir flask gunicorn

# Copy codebase
COPY models/ ./models/
COPY train_improved_tft.py .
COPY evaluate_improved_tft.py .
COPY serve_tft_model.py .

# Create directories for model storage and logs
RUN mkdir -p models/saved_models logs

# By default, expose port 5000 for the Flask API
EXPOSE 5000

# Create a non-root user to run the app
RUN useradd -m tftuser
RUN chown -R tftuser:tftuser /app
USER tftuser

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TZ=UTC

# Run the server using gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "serve_tft_model:init_app()"]

# Command for development (uncomment for local testing)
# CMD ["python", "serve_tft_model.py", "--host", "0.0.0.0", "--port", "5000"]