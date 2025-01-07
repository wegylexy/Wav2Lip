FROM nvcr.io/nvidia/pytorch:24.12-py3 AS base
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

FROM base AS build
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
ADD https://huggingface.co/mozi1924/wav2lip/resolve/main/wav2lip.pth?download=true checkpoints/wav2lip.pth
  # https://huggingface.co/Ftfyhh/wav2lip/resolve/main/wav2lip.pth?download=true
  # https://huggingface.co/Cong-HGMedia/wav2lip/resolve/main/wav2lip.pth?download=true
  # https://huggingface.co/gmk123/wav2lip/resolve/main/wav2lip.pth?download=true
ADD https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth /root/.cache/torch/hub/checkpoints/s3fd-619a316812.pth

FROM build AS publish
ENTRYPOINT [ "python", "/workspace/inference.py", "--checkpoint_path", "/workspace/checkpoints/wav2lip.pth" ]