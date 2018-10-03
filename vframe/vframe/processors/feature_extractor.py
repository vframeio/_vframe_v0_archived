"""
"""
# based on https://github.com/christiansafka/img2vec.git
# pip install torch==0.3.1 (0.4.0 does not work)
# pip install torchvision==0.2.1

import sys
import cv2 as cv
import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from vframe.utils import im_utils, logger_utils
from vframe.settings import types

log = logger_utils.Logger.getLogger()


class FeatureExtractor():

  def __init__(self, cuda=True, opt_net=types.PyTorchNet.RESNET18, layer='default'):
    """ Img2Vec
    :param cuda: If set to True, will run forward pass on GPU
    :param net: Enum type of requested model
    :param layer: String or Int depending on model.  See more docs: https://github.com/christiansafka/img2vec.git
    :param layer_output_size: Int depicting the output size of the requested layer
    """
    
    if opt_net == types.PyTorchNet.RESNET18:
      model_name = 'resnet-18'
    elif opt_net == types.PyTorchNet.ALEXNET:
      model_name = 'alexnet'

    self.device = torch.device("cuda" if cuda else "cpu")
    
    self.model, self.extraction_layer = self._get_model_and_layer(model_name, layer)

    self.model = self.model.to(self.device)

    self.model.eval()

    self.scaler = transforms.Scale((224, 224))
    self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])
    self.to_tensor = transforms.ToTensor()


  def _get_model_and_layer(self, model_name, layer):
    """ Internal method for getting layer from model
    :param model_name: model name such as 'resnet-18'
    :param layer: layer as a string for resnet-18 or int for alexnet
    :returns: pytorch model, selected layer
    """
    if model_name == 'resnet-18':
      model = models.resnet18(pretrained=True)
      if layer == 'default':
        layer = model._modules.get('avgpool')
        self.layer_output_size = 512
      else:
        layer = model._modules.get(layer)

      return model, layer

    elif model_name == 'alexnet':
      model = models.alexnet(pretrained=True)
      if layer == 'default':
        layer = model.classifier[-2]
        self.layer_output_size = 4096
      else:
        layer = model.classifier[-layer]

      return model, layer

    else:
      raise KeyError('Model %s was not found' % model_name)


  def extract(self, img, tensor=False, to_list=False):
    """ Get vector embedding from PIL image
    :param img: PIL Image
    :param tensor: If True, get_vec will return a FloatTensor instead of Numpy array
    :returns: Numpy ndarray
    """
    image = self.normalize(self.to_tensor(self.scaler(img))).unsqueeze(0).to(self.device)

    my_embedding = torch.zeros(1, self.layer_output_size, 1, 1)

    def copy_data(m, i, o):
      my_embedding.copy_(o.data)

    h = self.extraction_layer.register_forward_hook(copy_data)
    h_x = self.model(image)
    h.remove()

    if not tensor:
      tensor = my_embedding.numpy()[0, :, 0, 0]

    if to_list:
      tensor = tensor.tolist()
    
    return tensor


