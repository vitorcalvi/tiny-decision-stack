FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY . .
RUN pip install --upgrade pip && pip install '.[local]'

EXPOSE 8080
CMD ["uvicorn", "tiny_decision_stack.api:app", "--host", "0.0.0.0", "--port", "8080"]
