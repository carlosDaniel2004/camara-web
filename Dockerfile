FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias del sistema para compilar dlib y ejecutar opencv
RUN apt-get update && \
	apt-get install -y --no-install-recommends \
		build-essential \
		cmake \
		libgl1 \
		libglib2.0-0 \
		&& rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["gunicorn", "-b", "0.0.0.0:8000", "app.app:app"]
