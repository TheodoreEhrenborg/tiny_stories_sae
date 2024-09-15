#!/usr/bin/env bash
if ! [[ -f ~/.rye/env ]];
then
   tmp_dir=$(mktemp -d)
   cd $tmp_dir
   wget https://github.com/astral-sh/rye/releases/latest/download/rye-x86_64-linux.gz
   gunzip rye-x86_64-linux.gz
   chmod +x ./rye-x86_64-linux
   ./rye-x86_64-linux
fi
