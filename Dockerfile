FROM python:3.8-slim-buster

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    python3-dev \
    git \
    wget

WORKDIR /src

RUN git clone https://github.com/veichta/DeepSE-WF.git
WORKDIR /src/DeepSE-WF


RUN python -m pip install --upgrade pip && pip install -r requirements.txt

CMD ["bash"]

