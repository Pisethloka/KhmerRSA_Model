FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Layer 1: Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default python/python3 alias
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python

# Layer 2: Install Python packages
RUN pip3 install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    torch \
    transformers \
    soundfile \
    librosa \
    python-multipart \
    numpy

# Create app directory
WORKDIR /app

# Copy application files
COPY app.py /app/app.py

# Expose FastAPI port
EXPOSE 8000

# Run with uvicorn
ENTRYPOINT ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
