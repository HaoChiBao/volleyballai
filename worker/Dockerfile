# Cloud Run Job image — CPU pipeline only. AI inference calls Modal.
FROM python:3.11-slim

RUN apt-get update \
  && apt-get install -y --no-install-recommends ffmpeg \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY worker/requirements.txt /app/worker/requirements.txt
RUN pip install --no-cache-dir -r /app/worker/requirements.txt

COPY worker /app/worker

ENV PYTHONUNBUFFERED=1
# Cloud Run Jobs set JOB_ID / VIDEO_ID later; local poller mode is default entry.
CMD ["python", "-m", "worker"]
