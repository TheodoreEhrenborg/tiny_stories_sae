FROM ubuntu:22.04

RUN apt-get update && apt-get install -y curl \
    python3.10-venv python-is-python3 python3-pip

WORKDIR /deps
COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /code
