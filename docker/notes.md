# Docker notes

TODO

- improve permissions for deivce and xhost <https://towardsdatascience.com/real-time-and-video-processing-object-detection-using-tensorflow-opencv-and-docker-2be1694726e5>

xhost

- `xhost +local:docker` allows Docker access to Xserver
- `xhost -local:docker` disallows Docker access to Xserver
- `XSOCK=/tmp/.X11-unix`
- `XAUTH=/tmp/.docker.xauth`
- create xauth file: `xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -`
- run docker: `docker run -it --rm --device=/dev/video0 -e DISPLAY=$DISPLAY -v $XSOCK:$XSOCK -v $XAUTH:$XAUTH -e XAUTHORITY=$XAUTH`