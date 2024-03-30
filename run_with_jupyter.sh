#!/usr/bin/env sh
docker run -p 8888:8888 --gpus all -it -v $(pwd):/code $(cat docker_name) /bin/bash
