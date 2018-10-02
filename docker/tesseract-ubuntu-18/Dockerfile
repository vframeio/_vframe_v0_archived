# 
# Docker image for Tesseract 4 (and Leptonica) from source code
# https://github.com/tesseract-ocr/tesseract/wiki/Compiling#linux
# http://www.leptonica.org/source/README.html
# based on https://github.com/tesseract-shadow/tesseract-ocr-compilation/blob/master/Dockerfile
#

FROM ubuntu:18.04


ENV DOCKER_USER adam
RUN groupadd -r ${DOCKER_USER} \
  && useradd -r -g ${DOCKER_USER} ${DOCKER_USER}
USER ${DOCKER_USER}

ENV DEBIAN_FRONTEND noninteractive

# [ install dependencies ]

RUN apt-get update && apt-get install -y --\
	nano \
	pkg-config \
	python-dev \
	python-pip \
	python3-dev \
	python3-pip \
	python-tk \
	python3-tk \
	screen \
	wget \
	git


# [ install ocr ]

RUN apt install -y tesseract-ocr

# [ Install ZSH ]

RUN apt install -y zsh 
RUN git clone git://github.com/robbyrussell/oh-my-zsh.git /root/.oh-my-zsh
RUN cp /root/.oh-my-zsh/templates/zshrc.zsh-template ~/.zshrc
RUN chsh -s /bin/zsh


# [ Install Python packages ]


RUN pip3 install -U \
  setuptools \
  packaging \
  pyparsing \
  six \
  cython \
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
  pandas \
  tqdm \
  xmltodict \
  uwsgi \
  Flask \
  requests \
  python-dateutil


# [ ffmpeg ]

RUN apt-get update && \
 apt-get install -y --upgrade ffmpeg


# [ Download Tesseract data ]

ENV TESSDATA_PREFIX /usr/local/share/tessdata
RUN mkdir ${TESSDATA_PREFIX}
# osd	Orientation and script detection
RUN wget -O ${TESSDATA_PREFIX}/osd.traineddata https://github.com/tesseract-ocr/tessdata/raw/3.04.00/osd.traineddata
# equ	Math / equation detection
RUN wget -O ${TESSDATA_PREFIX}/equ.traineddata https://github.com/tesseract-ocr/tessdata/raw/3.04.00/equ.traineddata
# eng English
RUN wget -O ${TESSDATA_PREFIX}/eng.traineddata https://github.com/tesseract-ocr/tessdata/raw/4.00/eng.traineddata
# ara Arabic
RUN wget -O ${TESSDATA_PREFIX}/ara.traineddata https://github.com/tesseract-ocr/tessdata/raw/4.00/ara.traineddata
# other languages: https://github.com/tesseract-ocr/tesseract/wiki/Data-Files


# [ update config ]

RUN adduser --disabled-password --gecos "" docker && \
  echo 'docker ALL=NOPASSWD: ALL' >> /etc/sudoers && \
  su -c 'python3 -c "import matplotlib.pyplot"' docker && \
  echo 'ln -f /dev/null /dev/raw1394 2>/dev/null' >> /etc/bash.bashrc && \
  echo 'export PATH=/work/bin:/root/bin:${PATH}' >> /etc/bash.bashrc