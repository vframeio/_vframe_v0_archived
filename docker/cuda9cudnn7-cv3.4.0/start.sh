#!/bin/bash
# Startup script for docker image

xhost +local:docker

image=adamhrv/cuda8-cudnn6-opencv:3.4.0
docker images $image

# Jupyter notebook port
while getopts 'p:' flag; do
  case "${flag}" in
    p) port="${OPTARG}";;
    *) error "Unexpected option ${flag}" ;;
  esac
done

if [ ! -z "$port" ]; then
    echo "Port selected: $port"
else
    port="9090"
fi

docker_port="$port:$port"

# Get absolute path to vframe on local drive, could be refined
DOCKER_DIR=$(readlink -f "$0")
DOCKER_DIR_P1="$(dirname "$DOCKER_DIR")"
DOCKER_DIR_P2="$(dirname "$DOCKER_DIR_P1")"
vframe="$(dirname "$DOCKER_DIR_P2")"
echo $vframe

# --------------------------------------------------------

echo "
 __      ________ _____            __  __ ______ 
 \ \    / /  ____|  __ \     /\   |  \/  |  ____|
  \ \  / /| |__  | |__) |   /  \  | \  / | |__   
   \ \/ / |  __| |  _  /   / /\ \ | |\/| |  __|  
    \  /  | |    | | \ \  / ____ \| |  | | |____ 
     \/   |_|    |_|  \_\/_/    \_\_|  |_|______|
                                                 
                                            
Visual Forensics and Advanced Metadata Extraction

Stats:
$image
display=$DISPLAY
port=$docker_port

Jupyter:
jupyter notebook --port $port --ip 0.0.0.0 --allow-root --no-browser

"


# TODO
# Make ports accessible to jupyter
# maybe fix the unix display bug
#--workdir=$(pwd)
#--user $(id -u) \

nvidia-docker run -it --privileged \
	--hostname VFRAME-$(hostname|sed -e 's/ubuntu-//') \
	--volume /tmp/.X11-unix:/tmp/.X11-unix \
    --volume="/data_store:/data_store" \
    --volume="$vframe:/vframe" \
    --volume="/home/$USER:/home/$USER" \
    --volume="/etc/group:/etc/group:ro" \
    --volume="/etc/passwd:/etc/passwd:ro" \
    --volume="/etc/shadow:/etc/shadow:ro" \
    --volume="/etc/sudoers.d:/etc/sudoers.d:ro" \
    -e DISPLAY=unix$DISPLAY \
    -p $docker_port \
    -e 'DARKNET_NP=/opt/darknet_ah' \
    -e "PASSWORD=none" \
	-e "USER_HTTP=1" \
    $image "$@"
[ "$sshx" = "true" ] && kill %1 # kill backgrounded socat