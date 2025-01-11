FROM nvcr.io/nvidia/pytorch:24.12-py3 AS base
# Install ffmpeg
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

FROM base AS build
# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

# Pick one source
ADD https://huggingface.co/mozi1924/wav2lip/resolve/main/wav2lip.pth checkpoints/wav2lip.pth
# ADD https://huggingface.co/Ftfyhh/wav2lip/resolve/main/wav2lip.pth checkpoints/wav2lip.pth
# ADD https://huggingface.co/Cong-HGMedia/wav2lip/resolve/main/wav2lip.pth checkpoints/wav2lip.pth
# ADD https://huggingface.co/gmk123/wav2lip/resolve/main/wav2lip.pth checkpoints/wav2lip.pth

# Download face detection model
ADD https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth /root/.cache/torch/hub/checkpoints/s3fd-619a316812.pth

FROM build AS publish
ENTRYPOINT [ "python", "/workspace/inference.py", "--checkpoint_path", "/workspace/checkpoints/wav2lip.pth" ]