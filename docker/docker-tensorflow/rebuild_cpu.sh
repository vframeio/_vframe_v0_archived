#!/bin/bash
#nvidia-docker build --rm -t vframe $(readlink -f $(dirname $0))
docker build --rm -t vframe -f Dockerfile.cpu .
