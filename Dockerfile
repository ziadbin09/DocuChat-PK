FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 FASTEMBED_CACHE_PATH=/app/data/models
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY run.py .
EXPOSE 8000
CMD ["python", "run.py", "--host", "0.0.0.0", "--port", "8000"]
