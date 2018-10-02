#!/bin/bash
d=$(readlink -f $(dirname $0))
nvidia-docker build --rm -t vframe -f $d/Dockerfile.gpu $d