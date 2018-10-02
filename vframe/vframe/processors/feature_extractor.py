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
from torch.autograd import Variable

from vframe.utils import im_utils
from vframe.settings import types

class FeatureExtractor:

  def __init__(self, cuda=True, net=types.PyTorchNet.RESNET18, layer='default'):
    """FeatureExtractor
    :param cuda: If set to True, will run forward pass on GPU
    :param model: String name of requested model (alexnet or resnet-18)
    :param layer: String or Int depending on model.  See more docs: https://github.com/christiansafka/img2vec.git
    :param layer_output_size: Int depicting the output size of the requested layer
    """
    self.cuda = cuda
    self.net, self.extraction_layer = self._get_model_and_layer(net, layer)

    if self.cuda:
        self.net.cuda()

    self.net.eval()

    self.scaler = transforms.Resize((224, 224))
    self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])
    self.to_tensor = transforms.ToTensor()


  def _get_model_and_layer(self, opt_net, layer):
    """ Internal method for getting layer from model
    :param opt_net: model name such as 'resnet-18'
    :param layer: layer as a string for resnet-18 or int for alexnet
    :returns: pytorch model, selected layer
    """
    if opt_net == types.PyTorchNet.RESNET18:
        model = models.resnet18(pretrained=True)
        if layer == 'default':
            layer = model._modules.get('avgpool')
            self.layer_output_size = 512
        else:
            layer = model._modules.get(layer)

        return model, layer

    elif opt_net == types.PyTorchNet.ALEXNET:
        model = models.alexnet(pretrained=True)
        if layer == 'default':
            layer = model.classifier[-2]
            self.layer_output_size = 4096
        else:
            layer = model.classifier[-layer]

        return model, layer

    else:
        raise KeyError('Model %s was not found' % model_name)


  def extract(self, im, normalize=True, tensor=False):
    """ Get vector embedding from PIL image
    :param im: PIL Image or Numpy.ndarray
    :param normalize: If True, returns normalized feature vector
    :param tensor: If True, get_vec will return a FloatTensor instead of Numpy array
    :returns: Numpy ndarray
    """
    im = im_utils.ensure_pil(im)
    if self.cuda:
      image = Variable(self.normalize(self.to_tensor(self.scaler(im))).unsqueeze(0)).cuda()
    else:
      image = Variable(self.normalize(self.to_tensor(self.scaler(im))).unsqueeze(0))

    features = torch.zeros(self.layer_output_size)

    def copy_data(m, i, o):
        features.copy_(o.data)

    h = self.extraction_layer.register_forward_hook(copy_data)
    h_x = self.net(image)
    h.remove()

    if tensor:
      return features
    else:
      features = features.numpy()
      return features/np.linalg.norm(features)

  def get_vec(self, img, tensor=False):
        """ Get vector embedding from PIL image
        :param img: PIL Image
        :param tensor: If True, get_vec will return a FloatTensor instead of Numpy array
        :returns: Numpy ndarray
        """
        if self.cuda:
            image = Variable(self.normalize(self.to_tensor(self.scaler(img))).unsqueeze(0)).cuda()
        else:
            image = Variable(self.normalize(self.to_tensor(self.scaler(img))).unsqueeze(0))

        my_embedding = torch.zeros(self.layer_output_size)

        def copy_data(m, i, o):
            my_embedding.copy_(o.data)

        h = self.extraction_layer.register_forward_hook(copy_data)
        h_x = self.net(image)
        h.remove()

        if tensor:
            return my_embedding
        else:
            return my_embedding.numpy()