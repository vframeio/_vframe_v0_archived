#!/bin/bash
alias python='python3'
cd /vframe

# Set temporary IP route for syrian archive v2 site
echo "94.130.11.119 media.newsy.org newsy.org api.newsy.org" >> /etc/hosts

# Add the Darknet OpenCV/Numpy path to pythonpath
export PYTHONPATH="/opt/darknet_np/python":$PYTHONPATH

# start zsh default
/bin/zsh
