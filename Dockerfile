# ==============================================================================
# Safety AI - Production Dockerfile for Hugging Face Spaces & Cloud Deployments
# ==============================================================================

FROM python:3.10-slim

# Install system dependencies required by OpenCV, EasyOCR, and Video Streaming
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Configure non-root user according to Hugging Face Spaces guidelines (UID 1000)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PORT=7860

WORKDIR $HOME/app

# Pre-install lightweight CPU-only PyTorch & Torchvision (fast build, low memory)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy and install application Python requirements
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir easyocr

# Copy the complete project codebase and assets
COPY --chown=user:user . .

# Expose default Hugging Face Spaces port
EXPOSE 7860

# Start high-performance Flask server
CMD ["python", "server.py"]
