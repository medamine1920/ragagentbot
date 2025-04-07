FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y curl \
    libopenblas-dev \
    gfortran \
    build-essential \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app

# Copy requirements first for caching
COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire backend app
COPY . .

# Create static directory
RUN mkdir -p /app/static

# Set Python path to include /app and its subdirectories
ENV PYTHONPATH=/app

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
