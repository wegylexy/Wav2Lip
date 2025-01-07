FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime AS base
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

FROM base AS build
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install openh264
COPY . .
ADD https://huggingface.co/mozi1924/wav2lip/resolve/main/wav2lip.pth?download=true checkpoints/wav2lip.pth
  # https://huggingface.co/Ftfyhh/wav2lip/resolve/main/wav2lip.pth?download=true
  # https://huggingface.co/Cong-HGMedia/wav2lip/resolve/main/wav2lip.pth?download=true
  # https://huggingface.co/gmk123/wav2lip/resolve/main/wav2lip.pth?download=true

FROM build AS publish
ENTRYPOINT [ "python", "/workspace/inference.py", "--checkpoint_path", "/workspace/checkpoints/wav2lip.pth" ]