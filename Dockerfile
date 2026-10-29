# Serves the lexical + embedding scorer on CPU. The dataset and models are fetched and trained during
# the build, so the image never ships third-party data in this repository.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 USE_TF=0 HF_HOME=/app/.cache/huggingface
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu "torch==2.5.1+cpu" \
 && pip install --no-cache-dir -r requirements.txt

COPY sas ./sas
RUN python -m sas.data && python -m sas.evaluate --save-api

RUN useradd --create-home scorer && chown -R scorer /app
USER scorer
EXPOSE 8000
CMD ["uvicorn", "sas.api:app", "--host", "0.0.0.0", "--port", "8000"]
