# Dockerfile, Image, Container
# Container with python version 3.10
FROM ubuntu:20.04

COPY . /app

# change working directory
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y curl \
    && apt install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y wget \
    && apt-get install -y build-essential \
    && apt-get install -y python3.10 \
    && apt-get install -y python3.10-dev \
    && apt install -y python3.10-distutils \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && apt-get install -y python3-pip \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 \
    && apt-get install -y gnupg \
    && apt-get install -y sqlite3 \
    && apt-get install -y nvidia-cuda-toolkit \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /usr/share/dotnet \
    && rm -rf "$AGENT_TOOLSDIRECTORY"
    # && apt-get install -y --no-install-recommends nvidia-cuda-toolkit \
    # && pip install --upgrade pip setuptools wheel --no-cache-dir \

# upgrade pip and install pip packages
RUN pip install --upgrade pip && \
    pip install -r requirements.txt
    # Note: we had to merge the two "pip install" package lists here, otherwise
    # the last "pip install" command in the OP may break dependency resolution...

# run python program
#uvicorn language_model.main:app --reload
CMD ["uvicorn", "language_model.main:app","--host", "0.0.0.0", "--port", "8000", "--reload"]