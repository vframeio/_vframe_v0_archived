import sys

import cv2 as cv
import numpy as np

# load tf first to limit GPU RAM growth
#import tensorflow as tf
#from keras.backend.tensorflow_backend import set_session
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.3
#set_session(tf.Session(config=config))
# then import keras
import keras.applications
from keras.models import Model

from vframe.utils import im_utils, file_utils
from vframe.settings.types import KerasNet


class FeatureExtractor:
  
  def __init__(self, weights='imagenet', net=KerasNet.VGG16, gpu_ram=0.3, size=None):
    """Initialize Keras feature extractor
    :param weights: (str) name of the weights network
    :param net: (KerasNet) network type
    """

    from keras.preprocessing import image as keras_image
    self.keras_image = keras_image
    
    if net == KerasNet.XCEPTION:
      from keras.applications.xception import Xception
      from keras.applications.xception import preprocess_input
      self.model = Xception(weights=weights, include_top=False, pooling='avg')
      self.input_size = size if size is not None else (224,224)
    elif net == KerasNet.VGG16:
      from keras.applications.vgg16 import VGG16
      from keras.applications.vgg16 import preprocess_input
      self.model = VGG16(weights=weights, include_top=False, pooling='avg')
      self.input_size = size if size is not None else (224,224)
    elif net == KerasNet.VGG19:
      from keras.applications.vgg19 import VGG19
      from keras.applications.vgg19 import preprocess_input
      self.model = VGG19(weights=weights, include_top=False, pooling='avg')
      self.input_size = size if size is not None else (224,224)
    elif net == KerasNet.RESNET50:
      from keras.applications.resnet50 import ResNet50
      from keras.applications.resnet50 import preprocess_input
      self.model = ResNet50(weights=weights, include_top=False, pooling='avg')
      self.input_size = size if size is not None else (224,224)
    elif net == KerasNet.INCEPTIONV2:
      from keras.applications.inception_resnet_v2 import InceptionResNetV2
      from keras.applications.inception_resnet_v2 import preprocess_input
      self.model = InceptionResNetV2(weights=weights, include_top=False, pooling='avg')
      self.input_size = size if size is not None else (299,299)
    elif net == KerasNet.INCEPTIONV3:
      from keras.applications.inception_v3 import InceptionV3
      from keras.applications.inception_v3 import preprocess_input
      self.model = InceptionV3(weights=weights, include_top=False, pooling='avg')
      self.input_size = size if size is not None else (299,299)
    elif net == KerasNet.DENSENET121:
      from keras.applications.densenet import DenseNet121
      from keras.applications.densenet import preprocess_input
      self.model = DenseNet121(weights=weights, include_top=False, pooling='avg')
      self.input_size = size if size is not None else (224,224)
    elif net == KerasNet.DENSENET160:
      from keras.applications.densenet import DenseNet169
      from keras.applications.densenet import preprocess_input
      self.model = DenseNet169(weights=weights, include_top=False, pooling='avg')
      self.input_size = size if size is not None else (224,224)
    elif net == KerasNet.DENSENET201:
      from keras.applications.densenet import DenseNet201
      from keras.applications.densenet import preprocess_input
      self.model = DenseNet201(weights=weights, include_top=False, pooling='avg')
      self.input_size = size if size is not None else (224,224)
    elif net == KerasNet.NASNETLARGE:
      from keras.applications.nasnet import NASNetLarge
      from keras.applications.nasnet import preprocess_input
      self.model = NASNetLarge(weights=weights, include_top=False, pooling='avg')
      self.input_size = size if size is not None else (224,224)
    elif net == KerasNet.NASNETMOBILE:
      from keras.applications.nasnet import NASNetMobile
      from keras.applications.nasnet import preprocess_input
      self.model = NASNetMobile(weights=weights, include_top=False, pooling='avg')
      self.input_size = size if size is not None else (224,224)
    self.preprocess_input = preprocess_input


  def extract_fp(self, fp_im, normalize=True):
    """Loads image and generates"""
    im = cv.imread(fp_im)
    return self.extract(im, normalize=normalize)

  def extract(self, im, normalize=True):
    """Get vector embedding from Numpy image array
    :param im: Numpy.ndarray Image
    :param normalize: If True, returns normalized feature vector
    :returns: Numpy ndarray
    """
    
    # im = im_utils.ensure_np(im)
    im = cv.resize(im, self.input_size)  # force resize
    #im = keras_image.load_img(fp_im, target_size=(224, 224)) # convert np.ndarray
    x = self.keras_image.img_to_array(im)  # reshape to (3, 224, 224) 
    x = np.expand_dims(x, axis=0)  # expand to (1, 3, 224, 224)
    x = self.preprocess_input(x)
    feats = self.model.predict(x)[0]  # extract features
    #feats_arr = np.char.mod('%f', features) # convert to list
    if normalize:
      feats = feats/np.linalg.norm(feats)
    return feats