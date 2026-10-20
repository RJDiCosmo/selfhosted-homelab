FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# generated audio clips are written to /data, bind-mounted from the host
ENV OUTPUT_DIR=/data

EXPOSE 8030

CMD ["python", "main.py"]
