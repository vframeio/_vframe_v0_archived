FROM adamhrv/base:cuda9-cudnn7

# [ Install OpenCV 3.4.0 ]

WORKDIR /opt
RUN git clone https://github.com/opencv/opencv.git && \
  cd opencv && \
  git checkout 3.4.0

RUN cd /opt/opencv && \
  mkdir build && \
  cd build && \
  cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_C_EXAMPLES=OFF \
    -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D PYTHON_EXECUTABLE=/usr/bin/python3 \
  -D PYTHON_LIBRARY=/usr/lib/python3.5/config-3.5m-x86_64-linux-gnu/libpython3.5.so \
  -D PYTHON_NUMPY_INCLUDE_DIR=/usr/local/lib/python3.5/dist-packages/numpy/core/include/ \
  -D PYTHON_PACKAGES_PATH=/usr/local/lib/python3.5/dist-packages/ \
    -D WITH_TBB=ON \
    -D WITH_CUDA=ON \
    -D WITH_V4L=ON \
    -D WITH_QT=OFF \
    -D WITH_OPENGL=OFF \
    -D BUILD_PNG=1 \
    -D BUILD_JPEG=1 \
    -D WITH_FFMPEG=ON \
    -D BUILD_EXAMPLES=OFF .. && \
  make -j$(nproc) && \
  make install


# [ install tensorflow-gpu ]

RUN pip3 install tensorflow-gpu==1.5.0 && \
  pip2.7 install tensorflow-gpu==1.5.0


# [ Install Darknet ]

WORKDIR /opt
RUN git clone https://github.com/pjreddie/darknet darknet_pjreddie && \
  cd darknet_pjreddie && \
  sed -i 's/GPU=0/GPU=1/g' Makefile && \
  sed -i 's/CUDNN=0/CUDNN=1/g' Makefile && \
  sed -i 's/OPENCV=0/OPENCV=1/g' Makefile && \
  make


# [ Install YOLO V2 Python Interface ]

RUN pip2.7 install pkgconfig && \
  pip3 install pkgconfig

WORKDIR /opt
ENV YOLO3_4_PY YOLO3-4-Py
ENV DARKNET_HOME /opt/darknet_pjreddie
ENV LD_LIBRARY_PATH="/opt/darknet_pjreddie:${LD_LIBRARY_PATH}"

RUN git clone https://github.com/madhawav/YOLO3-4-Py $YOLO3_4_PY && \
  cd $YOLO3_4_PY && \
  export DARKNET_HOME=$DARKNET_HOME && \
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$DARKNET_HOME && \
  python3 setup.py build_ext --inplace


# [ new pips ]

RUN pip2 install imagehash && \
  pip3 install imagehash

RUN pip3 install http://download.pytorch.org/whl/cu90/torch-0.4.0-cp35-cp35m-linux_x86_64.whl && \
  pip3 install torchvision


# [ update sys ]

RUN ldconfig
RUN apt-get update

# [ if end docker node, Startup init ]

ENTRYPOINT /vframe/docker/entrypoint_gpu.sh