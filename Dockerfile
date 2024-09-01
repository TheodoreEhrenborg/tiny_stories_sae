FROM ubuntu:22.04

RUN apt-get update && apt-get install -y curl \
    python3.10-venv python-is-python3 python3-pip

WORKDIR /deps
COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /code

RUN pip install beartype==0.18.5 jaxtyping==0.2.33 tensorboard==2.17.1 coolname==2.2.0 pytest==8.3.2
