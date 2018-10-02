#!/bin/bash
nvidia-docker build --rm -t adamhrv/base:cuda8-cudnn6 $(readlink -f $(dirname $0))