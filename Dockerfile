FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir '.[serve]'
ENV FORGESIGHT_CHECKPOINT=/models/forgesight.pt
ENV FORGESIGHT_IMAGE_SIZE=256
CMD ["uvicorn", "forgesight.inference.service:app_from_env", "--factory", "--host", "0.0.0.0", "--port", "8000"]
