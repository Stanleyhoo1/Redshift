FROM python:3.12-slim

# MUJOCO_GL / PYOPENGL_PLATFORM: the onboard cameras (redshift/camera.py) render with OSMesa, i.e.
# software OpenGL, because the container has no GPU or display. MuJoCo reads MUJOCO_GL on import.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MUJOCO_GL=osmesa \
    PYOPENGL_PLATFORM=osmesa

# libosmesa6 is in Debian bookworm and trixie (if a newer base image lacks it, pin
# python:3.12-slim-trixie). It must be installed whenever MUJOCO_GL=osmesa is set: without the
# library `import mujoco` itself fails. If rendering fails at run time instead, only
# /api/camera answers 503 and the rest of the server keeps working.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libosmesa6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY redshift ./redshift

EXPOSE 8000
CMD ["uvicorn", "redshift.server:app", "--host", "0.0.0.0", "--port", "8000"]
