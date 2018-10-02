#!/bin/bash
nvidia-docker build --rm -t adamhrv/base:cuda9-cudnn7 $(readlink -f $(dirname $0))