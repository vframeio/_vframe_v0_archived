# Build notes for docker

Install docker for Ubuntu `Docker version 17.09.0-ce, build afdb6d4`

# Install nvidia, cudnn

Currently using 384.90


## Install nvidia-docker

Follow instructions from nvidia-docker github repo

```
# Install nvidia-docker and nvidia-docker-plugin
wget -P /tmp https://github.com/NVIDIA/nvidia-docker/releases/download/v1.0.1/nvidia-docker_1.0.1-1_amd64.deb
sudo dpkg -i /tmp/nvidia-docker*.deb && rm /tmp/nvidia-docker*.deb

# Test nvidia-smi
nvidia-docker run --rm nvidia/cuda nvidia-smi
```


- fix GCC error when building Darknet with GPU/CUDA
- <http://www.pittnuts.com/2016/07/geforce-gtx-1080-cuda-8-0-ubuntu-16-04-caffe/>
