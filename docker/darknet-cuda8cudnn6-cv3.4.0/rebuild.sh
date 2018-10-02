#!/bin/bash
nvidia-docker build --rm -t adamhrv/cuda8-cudnn6-darknet:3.4.0 $(readlink -f $(dirname $0))