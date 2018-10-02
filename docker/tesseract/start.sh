#!/bin/bash
xhost +local:docker

# nvidia-docker image
image=tesseract:latest
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

# Either for main UNAS drive or for mobile UNAS drive
if [ -d "/media/adam/unas2go" ]; then
    data_store="/media/adam/unas2go/data_store"
elif [ -d "/media/adam/unas2go" ]; then
    data_store="/media/adam/unas2go/data_store"
elif [ -d "/media/adam/ah3tb" ]; then
    data_store="/media/adam/ah3tb/data_store"
elif [ -d "/media/undisclosed/unas2go" ]; then
    data_store="/media/undisclosed/unas2go/data_store"
elif [ -d "/mnt/unas" ]; then
    data_store="/mnt/unas/data_store"
else
    echo '[-] No data_store available. Exiting.'
    exit 0
fi
echo $data_store
# Get absolute path to vframe on local drive, could be refined
DOCKER_DIR=$(readlink -f "$0")
DOCKER_DIR_P1="$(dirname "$DOCKER_DIR")"
DOCKER_DIR_P2="$(dirname "$DOCKER_DIR_P1")"
DOCKER_DIR_P3="$(dirname "$DOCKER_DIR_P2")"
vframe="$(dirname "$DOCKER_DIR_P3")"
echo $vframe

# --------------------------------------------------------

echo "
                                   
Tesseract OCR

Stats:
display=$DISPLAY
port=$docker_port

Jupyter:
jupyter notebook --port $port --ip 0.0.0.0 --allow-root --no-browser

"

docker run -it --privileged \
	--hostname OCR-$(hostname|sed -e 's/ubuntu-//') \
	--volume /tmp/.X11-unix:/tmp/.X11-unix \
    --volume="$data_store:/data_store" \
    --volume="$vframe:/vframe" \
    --volume="/home/$USER:/home/$USER" \
    --volume="/etc/group:/etc/group:ro" \
    --volume="/etc/passwd:/etc/passwd:ro" \
    --volume="/etc/shadow:/etc/shadow:ro" \
    --volume="/etc/sudoers.d:/etc/sudoers.d:ro" \
    -e DISPLAY=unix$DISPLAY \
    -p $docker_port \
	-e "USER_HTTP=1" \
    $image /bin/zsh
[ "$sshx" = "true" ] && kill %1 # kill backgrounded socat