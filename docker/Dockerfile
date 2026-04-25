FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# System deps for OpenCV and rasterio
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install the package
COPY . .
RUN pip install --no-cache-dir -e .

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "scripts/inference.py"]
