# 
# Docker image for Tesseract 4 (and Leptonica) from source code
# https://github.com/tesseract-ocr/tesseract/wiki/Compiling#linux
# http://www.leptonica.org/source/README.html
# based on https://github.com/tesseract-shadow/tesseract-ocr-compilation/blob/master/Dockerfile
#

FROM ubuntu:16.04

# [ install dependencies ]

RUN apt-get update && apt-get install -y \
	autoconf \
	autoconf-archive \
	automake \
	build-essential \
	checkinstall \
	cmake \
	g++ \
	git \
	libcairo2-dev \
	libcairo2-dev \
	libicu-dev \
	libicu-dev \
	libjpeg8-dev \
	libjpeg8-dev \
	libpango1.0-dev \
	libpango1.0-dev \
	libpng12-dev \
	libpng12-dev \
	libtiff5-dev \
	libtiff5-dev \
	libtool \
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
	xzgv \
	zlib1g-dev


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


# [ install Leptonica ]

ENV BASE_DIR /opt

# install leptonica

WORKDIR ${BASE_DIR}
RUN git clone https://github.com/DanBloomberg/leptonica.git
RUN mkdir leptonica/build && \
	cd leptonica/build && \
	cmake -DCMAKE_INSTALL_PREFIX=/usr/local -DBUILD_PROG=1 .. && \
	make -j$(nproc) && \
	make install


WORKDIR ${BASE_DIR}
RUN git clone https://github.com/tesseract-ocr/tesseract.git
RUN mkdir tesseract/build && \
	cd tesseract/build  && \
	PKG_CONFIG_PATH=/usr/local/lib/pkgconfig cmake \
	-DCMAKE_INSTALL_PREFIX=/usr/local \
	-DLeptonica_BUILD_DIR=/opt/leptonica/build \
	..  && \
	make -j$(nproc) && \
	make install && \
	export LD_LIBRARY_PATH=/opt/tesseract/build:$LD_LIBRARY_PATH && \
	ldconfig

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


# https://github.com/tesseract-ocr/tesseract/wiki/APIExample