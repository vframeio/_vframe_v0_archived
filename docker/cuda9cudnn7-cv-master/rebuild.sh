#!/bin/bash
nvidia-docker build --rm -t adamhrv/opencv:3.4.1 $(readlink -f $(dirname $0))