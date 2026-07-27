# GeoBench Copilot — Hugging Face Spaces (Docker SDK) image.
# CPU-only, no torch/CUDA: the backend only reads all_results.csv and calls the
# Claude API. rasterio's manylinux wheel bundles GDAL, so no system geo libs.
FROM python:3.11-slim

# HF Spaces runs as a non-root user (uid 1000) with a writable /home/user.
RUN useradd -m -u 1000 user
WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Backend code + bundled data; frontend files sit at the backend's PARENT
# (config.FRONTEND_DIR = BACKEND_DIR.parent), i.e. /app.
COPY backend /app/backend
COPY ["GeoBench Copilot.dc.html", "image-slot.js", "support.js", "/app/"]

# Per-run uploads land here; make it writable by the non-root user.
RUN mkdir -p /app/backend/.runs && chown -R user:user /app
USER user

ENV PORT=7860 \
    GEOBENCH_RUNS_DIR=/app/backend/.runs \
    PYTHONUNBUFFERED=1
EXPOSE 7860
WORKDIR /app/backend
# APP_PASSWORD and ANTHROPIC_API_KEY come from HF Space secrets at runtime.
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]
