FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_CONFIG_FILE=/dev/null
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgomp1 \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.docker.txt .
RUN python -m pip install --no-input --no-cache-dir --disable-pip-version-check -r requirements.docker.txt

COPY system/app ./system/app

EXPOSE 8090/tcp

CMD ["python", "system/app/bot_server.py"]
