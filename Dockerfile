# ==================== CPU-only Dockerfile ====================
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/opt/venv

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /app

# Copy requirements.txt
COPY requirements.txt .

# Install PyTorch CPU-only + other dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . /app

# Verify model file exists
RUN ls -lh resnet18_best.pth || echo "Warning: resnet18_best.pth not found"

# Default entrypoint
CMD ["python", "inference.py", "example_imgs/a1.jpg"]

