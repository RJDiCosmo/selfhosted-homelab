FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# bibles.db is mounted as a volume at /data/bibles.db
ENV DB_PATH=/data/bibles.db

EXPOSE 8000

CMD ["python", "main.py"]
