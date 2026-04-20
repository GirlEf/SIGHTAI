# ── SIGHTAI – Docker image ────────────────────────────────────────
# Base: Python 3.11 slim (TF 2.x compatible)
FROM python:3.11-slim

# System deps needed by TensorFlow / Pillow / scikit-image
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY . .

# Create directories that must exist at runtime
RUN mkdir -p saved_models

# Expose API port
EXPOSE 8000

# Run uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
