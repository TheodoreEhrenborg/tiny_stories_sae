FROM ubuntu:22.04

RUN apt-get update && apt-get install -y wget

RUN echo "source ~/.rye/env" >> ~/.bashrc

WORKDIR /code
