#!/bin/bash
# Startup script for docker image

xhost +local:docker

image=adamhrv/base:cuda8-cudnn6
docker images $image

echo "

 _   _ _   _ ____ ___ ____   ____ _     ___  ____  _____ ____  
| | | | \ | |  _ \_ _/ ___| / ___| |   / _ \/ ___|| ____|  _ \ 
| | | |  \| | | | | |\___ \| |   | |  | | | \___ \|  _| | | | |
| |_| | |\  | |_| | | ___) | |___| |__| |_| |___) | |___| |_| |
 \___/|_| \_|____/___|____/ \____|_____\___/|____/|_____|____/ 
                                                               
Docker base image for cuda8-cudnn6
"
# Start the docker container with access to USB devices
# Make ports accessible to Jupyter

nvidia-docker run -it --rm --privileged \
	-e DISPLAY=$DISPLAY \
	--volume /tmp/.X11-unix:/tmp/.X11-unix \
	--user docker \
	--volume "/data_store:/data_store" \
	-e "USER_HTTP=1" $image "/bin/zsh"
[ "$sshx" = "true" ] && kill %1 # kill backgrounded socat