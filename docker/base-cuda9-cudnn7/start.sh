#!/bin/bash
# Startup script for docker image

image=base:cuda9-cudnn7
docker images $image

echo "

 _   _ _   _ ____ ___ ____   ____ _     ___  ____  _____ ____  
| | | | \ | |  _ \_ _/ ___| / ___| |   / _ \/ ___|| ____|  _ \ 
| | | |  \| | | | | |\___ \| |   | |  | | | \___ \|  _| | | | |
| |_| | |\  | |_| | | ___) | |___| |__| |_| |___) | |___| |_| |
 \___/|_| \_|____/___|____/ \____|_____\___/|____/|_____|____/ 
                                                               
Docker base image for cuda9-cudnn7
"
# Start the docker container with access to USB devices
# Make ports accessible to Jupyter

nvidia-docker run -it --rm --privileged \
	-h undisclosed-$(hostname|sed -e 's/ubuntu-//') \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-e DISPLAY=$DISPLAY \
	-e "PASSWORD=none" \
	-e "USER_HTTP=1" $image "$@"
[ "$sshx" = "true" ] && kill %1 # kill backgrounded socat