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


# --------------------------------------------------------

echo "
 __      ________ _____            __  __ ______ 
 \ \    / /  ____|  __ \     /\   |  \/  |  ____|
  \ \  / /| |__  | |__) |   /  \  | \  / | |__   
   \ \/ / |  __| |  _  /   / /\ \ | |\/| |  __|  
    \  /  | |    | | \ \  / ____ \| |  | | |____ 
     \/   |_|    |_|  \_\/_/    \_\_|  |_|______|
                                                 
                                            
Visual Forensics and Metadata Extraction

image:    $image
display:  $DISPLAY
port:     $docker_port

Jupyter:
jupyter notebook --port $port --ip 0.0.0.0 --allow-root --no-browser

"


# TODO
# Make ports accessible to jupyter
# maybe fix the unix display bug
#--workdir=$(pwd)
#--user $(id -u) \
PASS='adam'
# --volume="/home/$USER:/home/$USER" \

nvidia-docker run -it --privileged \
  --hostname VFRAME-$port \
  --volume /tmp/.X11-unix:/tmp/.X11-unix \
    --volume="/data_store:/data_store" \
    --volume="$vframe:/vframe" \
    --volume="/etc/group:/etc/group:ro" \
    --volume="/etc/passwd:/etc/passwd:ro" \
    --volume="/etc/shadow:/etc/shadow:ro" \
    --volume="/etc/sudoers.d:/etc/sudoers.d:ro" \
    -e DISPLAY=unix$DISPLAY \
    -p $docker_port \
    -e "PYTHONPATH=/opt/YOLO3-4-Py/:$PYTHONPATH" \
    -e "PASSWORD=none" \
    -u `id -u $USER` \
    -e "USER_HTTP=1" \
    -e "$PASS\n$PASS" \
      $image "/vframe/docker/entrypoint_gpu.sh"
[ "$sshx" = "true" ] && kill %1 # kill backgrounded socat