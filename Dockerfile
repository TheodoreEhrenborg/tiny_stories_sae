FROM ubuntu:22.04

RUN apt-get update && apt-get install -y curl

RUN apt-get update && apt-get install -y python3.10-venv python-is-python3

RUN apt-get update && apt-get install -y python3-pip

COPY requirements.txt .

RUN pip install -r requirements.txt

WORKDIR /code
