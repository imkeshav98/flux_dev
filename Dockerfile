# Use a base image with CUDA 12.4 support
FROM nvidia/cuda:12.4.0-base-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_TOKEN=your_hugging_face_token

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create a directory for the application and copy server.py
WORKDIR /app
COPY server.py /app/

# Expose port 8000
EXPOSE 8000

# Command to run the server
CMD ["python", "server.py"]