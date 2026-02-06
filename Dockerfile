# Base image with CUDA runtime for GPU support
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11 and basic tools
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3.11-distutils curl git wget build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH
ENV PATH="/root/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Default command: bash shell
CMD ["bash"]
