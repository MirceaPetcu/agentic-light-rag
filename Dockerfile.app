# Multi-Agent LightRAG API Dockerfile
FROM python:3.12-slim

WORKDIR /app    

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    # For textract file processing
    antiword \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

RUN pip install --no-cache-dir "pip<24.1"

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install textract for file processing
RUN pip install --no-cache-dir textract

# Install LightRAG from local submodule
COPY light_rag /app/light_rag
RUN pip install --no-cache-dir -e /app/light_rag

# Copy application code
COPY *.py /app/

# Create storage directory
RUN mkdir -p /app/rag_storage

# Expose port
EXPOSE 8000

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV HOST=0.0.0.0
ENV PORT=8000

# Run the application
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]