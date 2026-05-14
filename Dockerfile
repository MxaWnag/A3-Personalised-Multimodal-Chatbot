FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt /app/requirements.txt
# sentence-transformers pulls torch; default PyPI torch = CUDA + huge nvidia-* wheels (~GB download).
# Install CPU torch first so Docker builds stay small unless you use an nvidia/cuda base + GPU torch.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch>=2.1,<3" \
    && pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

EXPOSE 8000 8501
