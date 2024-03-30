#!/usr/bin/env sh
docker run --gpus all -it -v $(pwd):/code $(cat docker_name) /bin/bash
