# d3 GUI in a Linux container (GPU). Model weights come from the host HF cache mount, not the image.
FROM python:3.12-slim
# gcc: Triton (used by flash-linear-attention) compiles a small launcher at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
WORKDIR /app
COPY gui.py gui.html ./
ENV PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1
EXPOSE 7860
CMD ["python", "gui.py", "--host", "0.0.0.0"]
