# Dockerfile

FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    tesseract-ocr libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip config set global.index-url https://pypi.org/simple
#RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --timeout=600 --no-cache-dir -r requirements.txt

COPY backend/APP /app
COPY frontend /app/frontend

EXPOSE 8000
EXPOSE 8501

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]