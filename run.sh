#!/usr/bin/env sh
docker run -it -v $HOME/.cache/huggingface:/root/.cache/huggingface -v $(pwd):/code $(cat docker_name) /bin/bash
