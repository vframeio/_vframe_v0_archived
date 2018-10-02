#!/bin/bash
nvidia-docker build --rm -t adamhrv/cuda9-cudnn7-opencv:3.4.0 $(readlink -f $(dirname $0))