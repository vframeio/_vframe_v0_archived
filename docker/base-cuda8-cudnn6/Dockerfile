FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04

# Docker base environment
# Includes most dependences for deep learning, computer vision

ENV DEBIAN_FRONTEND noninteractive

MAINTAINER Adam Harvey

# Ubuntu 16.04 + Cuda 8.0 + CUDNN 6.0 + Python2.7 + Python3.5
# Using Nvidia driver 367.51 and nvidia-docker
# Install Nvidia driver 367.51: 
#   add-apt-repository ppa:graphics-drivers/ppa
#   apt-get install nvidia-367

# [ environment paths ]

RUN echo export CUDA_HOME=/usr/local/cuda/ >> /etc/bash.bashrc
RUN echo export PATH=/root/bin/:${CUDA_HOME}/bin:${PATH} >> /etc/bash.bashrc
RUN echo export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/lib" >> /etc/bash.bashrc

# --------------------------------------------------------

# [ Update and upgrade Ubuntu ]

RUN apt-get update

RUN apt-get install -y --no-install-recommends software-properties-common && \
  add-apt-repository "deb http://archive.ubuntu.com/ubuntu $(lsb_release -sc) multiverse"
RUN apt-get update && \
  apt-get upgrade -y


# --------------------------------------------------------

# [ System dependencies ]

RUN apt-get install -y \
  ant \
  bc \
  build-essential \
  checkinstall \
  cmake \
  curl \
  default-jdk \
  doxygen \
  git \
  gfortran \
  gir1.2-gst-plugins-base-0.10 \
  gir1.2-gstreamer-0.10 \
  imagemagick \
  iproute2 \
  mediainfo \
  nano \
  nginx \
  pkg-config \
  protobuf-compiler \
  python-cffi \
  python-dev \
  python-magic \
  python-h5py \
  python-numpy \
  python-pip \
  python-pythonmagick \
  python-tk \
  python-qt4 \
  python-yaml \
  python-xtermcolor \
  qt5-default \
  rsync \
  supervisor \
  screen \
  sphinx-common \
  texlive-latex-extra \
  tesseract-ocr \
  x264 \
  v4l-utils \
  vim \
  unzip \
  vlc \
  wget \
  xauth \
  yasm \
  youtube-dl \
  zip \
  zlib1g-dev
  
RUN apt-get update 

RUN apt-get install -y  \
  libatlas-base-dev \
  libavcodec-dev \
  libavformat-dev \
  libcurl3-dev \
  libdc1394-22-dev \
  libeigen3-dev \
  libfaac-dev \
  libffi-dev \
  libgflags-dev \
  libfreetype6-dev \
  libgoogle-glog-dev \
  libgstreamer-plugins-base0.10-0 \
  libgstreamer-plugins-base0.10-dev \
  libgstreamer0.10-0 \
  libgstreamer0.10-dev \
  libgtk2.0-dev \
  libhdf5-dev \
  libhdf5-serial-dev \
  libjasper-dev \
  libjpeg-dev \
  libjpeg8-dev \
  libleveldb-dev \
  liblmdb-dev \
  libmp3lame-dev \
  libopencore-amrnb-dev \
  libopencore-amrwb-dev \
  libopenexr-dev \
  libpng12-dev \
  libprotobuf-dev \
  libqt4-dev \
  libqt4-opengl-dev \
  libreadline-dev \
  libsnappy-dev \
  libssl-dev \
  libswscale-dev \
  libtbb-dev \
  libtheora-dev \
  libtiff5-dev \
  libvtk5-qt4-dev \
  libv4l-dev \
  libxine2-dev \
  libvorbis-dev \
  libx264-dev \
  libatlas-base-dev \
  libgphoto2-dev \
  libxvidcore-dev \
  libzmq3-dev && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
  apt-get install -y --no-install-recommends libboost-all-dev && \
  apt-get install -y pv locate inkscape && \
  updatedb


# [ install pip2/3 ]

RUN apt-get update
RUN apt-get install -y python-dev python-pip python3-dev python3-pip
RUN pip2 install -U pip
RUN pip3 install -U pip


# [ add Github creds ]

RUN git config --global user.name "docker" && \
  git config --global user.email "docker@docker.com"


# [ Install Python2.7 packages ]

RUN pip2.7 install -U \
  setuptools \
  packaging \
  pyparsing \
  six \
  cython \
  svgwrite \
  numpy \
  sklearn \
  scikit-image \
  scikit-learn \
  imutils \
  Pillow \
  matplotlib \
  argparse \
  jupyter \
  scipy \
  easydict \
  click \
  pandas \
  ipdb \
  python-osc \
  tqdm \
  xmltodict \
  librosa \
  uwsgi \
  Flask \
  requests \
  bcolz \
  sympy

# [ Install Python3 packages ]

RUN pip3 install -U \
  setuptools \
  packaging \
  pyparsing \
  six \
  cython \
  svgwrite \
  numpy \
  sklearn \
  scikit-image \
  scikit-learn \
  imutils \
  Pillow \
  matplotlib \
  argparse \
  jupyter \
  scipy \
  easydict \
  click \
  pandas \
  ipdb \
  python-osc \
  tqdm \
  xmltodict \
  librosa \
  uwsgi \
  Flask \
  requests \
  python-dateutil \
  bcolz \
  sympy


# [ ffmpeg ]

RUN apt-get update && \
   apt-get install -y --upgrade ffmpeg


# [ additional ]

RUN apt-get update -y

RUN apt-get install \
  libcanberra-gtk-module \
  python3-tk -y


# [ define docker user ]

ENV DOCKER_USER docker

# [ Install ZSH, in home directory ]

#RUN adduser --disabled-password --gecos "" ${DOCKER_USER} && \
#    echo '${DOCKER_USER} ALL=NOPASSWD: ALL' >> /etc/sudoers

RUN useradd -d /home/${DOCKER_USER} -ms /bin/bash -g root -G sudo -p ${DOCKER_USER} ${DOCKER_USER}

RUN apt install -y zsh 
RUN git clone git://github.com/robbyrussell/oh-my-zsh.git /home/${DOCKER_USER}/.oh-my-zsh
RUN cp /home/${DOCKER_USER}/.oh-my-zsh/templates/zshrc.zsh-template /home/${DOCKER_USER}/.zshrc
RUN chsh -s $(which zsh)


# [ update config ]

RUN su -c 'python -c "import matplotlib.pyplot"' ${DOCKER_USER} && \
    python -c 'import matplotlib.pyplot' && \
    echo 'ln -f /dev/null /dev/raw1394 2>/dev/null' >> /etc/bash.bashrc && \
    echo 'export PATH=/work/bin:/root/bin:${PATH}' >> /etc/bash.bashrc

WORKDIR /home/${DOCKER_USER}