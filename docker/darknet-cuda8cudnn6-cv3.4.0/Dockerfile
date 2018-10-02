FROM adamhrv/cuda8-cudnn6-cv3.4.0

# -----------------------------------------------------------------
#
# Darknet container for training
#
# -----------------------------------------------------------------


# [ Install Darknet ]

WORKDIR /opt
RUN git clone https://github.com/pjreddie/darknet darknet_pjreddie && \
  cd darknet_pjreddie && \
  sed -i 's/GPU=0/GPU=1/g' Makefile && \
  sed -i 's/CUDNN=0/CUDNN=1/g' Makefile && \
  sed -i 's/OPENCV=0/OPENCV=1/g' Makefile && \
  make


# [ pyyolo interface Yolo2/Yolo3 ]

WORKDIR /opt
RUN git clone --recursive https://github.com/rayhou0710/pyyolo.git && \
  cd pyyolo && \
  make && \
  rm -rf build && \
  python3 setup_gpu.py build && \
  python3 setup_gpu.py install

