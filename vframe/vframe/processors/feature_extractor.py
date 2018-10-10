"""
Feature extractor in PyTorch with CUDA
"""
# based on https://github.com/christiansafka/img2vec.git
# see https://github.com/christiansafka/img2vec/issues/7

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

  def __init__(self, cuda=True, opt_net=types.PyTorchNet.RESNET18):
    """FeatureExtractor
    :param cuda: true for GPU false for CPU
    :param opt_net: Enum type of requested model
    """
    
    self.opt_net = opt_net

    self.device = torch.device("cuda" if cuda else "cpu")
    
    # set layer size and type
    if self.opt_net == types.PyTorchNet.RESNET18:
      # (avgpool): AvgPool2d(kernel_size=7, stride=1, padding=0)
      self.model = models.resnet18(pretrained=True)
      self.layer = self.model.avgpool
      self.layer_output_size = 512

    elif self.opt_net == types.PyTorchNet.ALEXNET:
      # (4): Linear(in_features=4096, out_features=4096, bias=True)
      self.model = models.alexnet(pretrained=True)
      self.layer = self.model.classifier[-2]
      self.layer_output_size = 4096

    elif self.opt_net == types.PyTorchNet.VGG16:
      # (3): Linear(in_features=4096, out_features=4096, bias=True)
      self.model = models.vgg16(pretrained=True)
      self.layer = self.model.classifier[3]
      self.layer_output_size = 4096

    elif self.opt_net == types.PyTorchNet.VGG19:
      # (3): Linear(in_features=4096, out_features=4096, bias=True)
      self.model = models.vgg19(pretrained=True)
      self.layer = self.model.classifier[3]
      self.layer_output_size = 4096

    # convert model to gpu/cpu
    self.model = self.model.to(self.device)

    # unsure
    self.model.eval()

    # image preprocessing
    # TODO double check that all networks have same values
    self.scaler = transforms.Resize((224, 224))
    self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    self.to_tensor = transforms.ToTensor()


  def extract(self, im_pil, to_list=False, normalize=False):
    """Get feature vector from PIL image
    :param im_pil: PIL Image
    :param tensor: If True, get_vec will return a FloatTensor instead of Numpy array
    :returns: Numpy ndarray
    """
    im_pth = self.normalize(self.to_tensor(self.scaler(im_pil))).unsqueeze(0).to(self.device)

    if self.opt_net == types.PyTorchNet.RESNET18:
      vec_pth = torch.zeros(1, self.layer_output_size, 1, 1)
    else:
      vec_pth = torch.zeros(1, self.layer_output_size)

    def copy_data(m, i, o):
      vec_pth.copy_(o.data)

    h = self.layer.register_forward_hook(copy_data)
    h_x = self.model(im_pth)
    h.remove()

    if self.opt_net == types.PyTorchNet.RESNET18:
      vec_np = vec_pth.numpy()[0, :, 0, 0]
    else:
      vec_np = vec_pth.numpy()[0, :]

    if normalize:
      vec_np = vec_np/np.linalg.norm(vec_np)

    if to_list:
      return vec_np.tolist()
    else:
      return vec_np

