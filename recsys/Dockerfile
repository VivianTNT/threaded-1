FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface

WORKDIR /app

COPY recsys/requirements-api.txt /tmp/requirements-api.txt
RUN pip install -r /tmp/requirements-api.txt

COPY recsys /app/recsys
RUN chmod +x /app/recsys/start-api.sh

EXPOSE 8000

CMD ["/app/recsys/start-api.sh"]
