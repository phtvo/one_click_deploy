# syntax=docker/dockerfile:1.13-labs
FROM ghcr.io/ggml-org/llama.cpp:server-cuda

RUN apt-get update && apt-get install -y git build-essential \
python3.11 python3-pip gcc

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
RUN update-alternatives --set python3 /usr/bin/python3.11

# =============    
# Install depencencies
RUN python3 -m pip install --upgrade pip


# Update clarifai package so we always have latest protocol to the API. Everything should land in /venv
COPY --link requirements.txt /home/nonroot/requirements.txt
RUN ["pip", "install", "--no-cache-dir", "-r", "/home/nonroot/requirements.txt"]
RUN ["pip", "show", "clarifai"]

RUN ln -s /app/llama-server /usr/local/bin/llama-server
# Set the NUMBA cache dir to /tmp
# Set the TORCHINDUCTOR cache dir to /tmp
# The CLARIFAI* will be set by the templaing system.
ENV NUMBA_CACHE_DIR=/tmp/numba_cache \
    TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache \
    HOME=/tmp \
    DEBIAN_FRONTEND=noninteractive

#####
# Copy the files needed to download
#####
# This creates the directory that HF downloader will populate and with nonroot:nonroot permissions up.
COPY --chown=nonroot:nonroot downloader/unused.yaml /home/nonroot/main/1/checkpoints/.cache/unused.yaml


#####
# Download checkpoints if config.yaml has checkpoints.when = "build"
COPY --link=true config.yaml /home/nonroot/main/
RUN ["python3", "-m", "clarifai.cli", "model", "download-checkpoints", "--model_path", "/home/nonroot/main", "--out_path", "/home/nonroot/main/1/checkpoints", "--stage", "build"]
#####

# Copy in the actual files like config.yaml, requirements.txt, and most importantly 1/model.py
# for the actual model.
# If checkpoints aren't downloaded since a checkpoints: block is not provided, then they will
# be in the build context and copied here as well.
COPY --link=true 1 /home/nonroot/main/1
# At this point we only need these for validation in the SDK.
COPY --link=true requirements.txt config.yaml /home/nonroot/main/

# Add the model directory to the python path.
ENV PYTHONPATH=${PYTHONPATH}:/home/nonroot/main \
    CLARIFAI_PAT=${CLARIFAI_PAT} \
    CLARIFAI_USER_ID=${CLARIFAI_USER_ID} \
    CLARIFAI_RUNNER_ID=${CLARIFAI_RUNNER_ID} \
    CLARIFAI_NODEPOOL_ID=${CLARIFAI_NODEPOOL_ID} \
    CLARIFAI_COMPUTE_CLUSTER_ID=${CLARIFAI_COMPUTE_CLUSTER_ID} \
    CLARIFAI_API_BASE=${CLARIFAI_API_BASE:-https://api.clarifai.com} \
    CLARIFAI_NUM_THREADS=${CLARIFAI_NUM_THREADS}


RUN ln -s /usr/bin/python3 /usr/bin/python
# Finally run the clarifai entrypoint to start the runner loop and local dev server.
# Note(zeiler): we may want to make this a clarifai CLI call.
ENTRYPOINT ["python3", "-m", "clarifai.runners.server"]
CMD ["--model_path", "/home/nonroot/main"]
#############################
