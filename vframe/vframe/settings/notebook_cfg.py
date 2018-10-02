"""
Jupyter notebook settings
"""

import os
from os.path import join


# General paths
DATA_STORE = '/data_store/'
DIR_APP = '/data_store/apps/vframe'
DIR_MODELS = join(DIR_APP,'models')

# General paths
DATA_BODEGA = '/vframe/data_bodega/'

# ----------------------------------
# DNN model paths

# YOLO/Darknet Models
DIR_OPENCV_DATA = join(DIR_MODELS,'darknet')

# OpenCV Data
DIR_OPENCV_DATA = join(DIR_MODELS,'opencv')

# Caffe
DIR_CAFFE_DATA = join(DIR_MODELS,'caffe')
CAFFE_SSD_FACE_PROTO = join(DIR_CAFFE_DATA,'opencv_dnn','opencv_face_detector.prototxt')
CAFFE_SSD_FACE_MODEL = join(DIR_CAFFE_DATA,'opencv_dnn','opencv_face_detector.caffemodel')

# DLIB
DIR_DLIB_DATA = join(DIR_MODELS,'dlib')
dlib_hog_dat = 'shape_predictor_68_face_landmarks.dat'
DLIB_HOG_DATA = join(DIR_DLIB_DATA,dlib_hog_dat)
dlib_cnn_dat = 'mmod_human_face_detector.dat'
DLIB_CNN_DATA = join(DIR_DLIB_DATA,dlib_cnn_dat)

#DLIB_FACEREC_DAT = join(DIR_DLIB_DATA,dlib_hog_dat)
#dlib_cnn_dat = 'shape_predictor_68_face_landmarks.dat'
#dlib_facerec_dat = 'shape_predictor_68_face_landmarks.dat'