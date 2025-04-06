FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libopenblas-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#copy the application
COPY . .

#set python path
ENV PYTHONPATH=/app

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

