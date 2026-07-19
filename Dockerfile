FROM mambaorg/micromamba:1.5.6-bullseye-cuda-12.1.1

LABEL maintainer="LatentSync"
LABEL description="Docker image for LatentSync API using mamba"

USER root

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Switch to the mamba user
USER $MAMBA_USER

# Copy requirements file
COPY --chown=$MAMBA_USER:$MAMBA_USER requirements.txt /app/

# Install python and pytorch using mamba, then pip install the rest
RUN micromamba install -y -n base -c pytorch -c nvidia -c conda-forge \
    python=3.10 \
    pytorch=2.5.1 \
    torchvision=0.20.1 \
    torchaudio \
    pytorch-cuda=12.1 \
    && micromamba clean --all --yes

# Activate mamba and pip install requirements
ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY --chown=$MAMBA_USER:$MAMBA_USER . /app/

EXPOSE 9880

CMD ["python", "api.py"]
