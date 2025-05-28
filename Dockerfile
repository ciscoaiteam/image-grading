# Dockerfile
FROM python:3.9-slim

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends supervisor && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY src /app/src

# Copy trained model weights to root
COPY src/fastapi/model.pth /app/model.pth

# Copy Supervisor configuration into image
COPY supervisord.conf /etc/supervisor/supervisord.conf

# Create log directory for Supervisor
RUN mkdir -p /app/logs

# Expose ports for FastAPI and Streamlit
EXPOSE 8000 8501

# Start Supervisor (manages FastAPI and Streamlit)
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/supervisord.conf"]
