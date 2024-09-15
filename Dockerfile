FROM ubuntu:22.04

WORKDIR /rye_download

RUN apt-get update && apt-get install -y curl
RUN apt-get install -y wget

RUN wget https://github.com/astral-sh/rye/releases/latest/download/rye-x86_64-linux.gz

RUN gunzip rye-x86_64-linux.gz

RUN chmod +x ./rye-x86_64-linux

RUN echo "source ~/.rye/env" >> ~/.bashrc

WORKDIR /code
