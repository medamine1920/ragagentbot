# Dockerfile

FROM python:3.11-slim

# Install system dependencies (tesseract, libsm6, etc.)
RUN apt-get update && apt-get install -y \
    tesseract-ocr libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# FIRST: Copy only requirements and install dependencies (cache friendly)
COPY requirements.txt .

RUN pip config set global.index-url https://pypi.org/simple
RUN pip install --timeout=600 --no-cache-dir -r requirements.txt

# THEN: Copy the actual application code
COPY backend/APP /app
#COPY frontend /app/frontend
#COPY app .
# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000
EXPOSE 8501

# Start FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
