import os
from os.path import join


DIR_MODELS = join(DIR_APP_VFRAME,'models')

# Frameworks
DIR_MODELS_BRISQUE = join(DIR_MODELS, 'brisque')
DIR_MODELS_CAFFE = join(DIR_MODELS,'caffe')
DIR_MODELS_DLIB = join(DIR_MODELS, 'dlib')
DIR_MODELS_DARKNET = join(DIR_MODELS,'darknet')
DIR_MODELS_KERAS = join(DIR_MODELS, 'keras')
DIR_MODELS_MXNET = join(DIR_MODELS,'mxnet')
DIR_MODELS_OPENCV = join(DIR_MODELS,'opencv')
DIR_MODELS_PYTORCH = join(DIR_MODELS,'pytorch')
DIR_MODELS_TF = join(DIR_MODELS,'tensorflow')
DIR_MODELS_TORCH = join(DIR_MODELS,'torch')

# Caffe networks: move to YAML
DIR_CAFFE_PLACES365 = join(DIR_MODELS_CAFFE,'places365')
DIR_CAFFE_PLACES365_GOOGLENET = join(DIR_CAFFE_PLACES365,'googlenet_places365')
DIR_CAFFE_PLACES365_GOOGLENET_PROTO = join(
	DIR_CAFFE_PLACES365_GOOGLENET,'deploy_googlenet_places365.prototxt')
DIR_CAFFE_PLACES365_GOOGLENET_MODEL = join(
	DIR_CAFFE_PLACES365_GOOGLENET,'deploy_googlenet_places365.caffemodel')
DIR_CAFFE_PLACES365_GOOGLENET_META = join(
	DIR_CAFFE_PLACES365,'metadata/categories_places365.txt')
CAFFE_PLACES365_GOOGLENET_NUM_CLASSES = 365
CAFFE_PLACES365_GOOGLENET_MEAN = (104, 117, 123)
CAFFE_PLACES365_GOOGLENET_SIZE = (224, 224)
