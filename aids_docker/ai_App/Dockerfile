# Use official Python slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    tmux \
    wget \
    unzip \
    btop \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install Python packages
COPY requirements.txt .
RUN pip install -r requirements.txt
# RUN pip install \
#     opencv-python \
#     ultralytics \
#     psycopg2-binary \
#     numpy \
#     matplotlib

# Default command
CMD [ "bash" ]
