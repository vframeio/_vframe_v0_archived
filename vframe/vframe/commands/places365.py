"""
Generates Places365 metadata using OpenCV's DNN module
"""
import click

from vframe.settings import types
from vframe.utils import click_utils
from vframe.settings import vframe_cfg as cfg

from cli_vframe import processor


@click.command('gen_places365')
@click.option('-d', '--disk', 'opt_disk',
  default=click_utils.get_default(types.DataStore.SSD),
  type=cfg.DataStoreVar,
  show_default=True,
  help=click_utils.show_help(types.DataStore))
@click.option('--density', 'opt_density',
  default=click_utils.get_default(types.KeyframeMetadata.EXPANDED),
  show_default=True,
  type=cfg.KeyframeMetadataVar,
  help=click_utils.show_help(types.KeyframeMetadata))
@click.option('--size', 'opt_size',
  type=cfg.ImageSizeVar,
  default=click_utils.get_default(types.ImageSize.MEDIUM),
  help=click_utils.show_help(types.ImageSize))
@processor
@click.pass_context
def cli(ctx, sink, opt_disk, opt_density, opt_size):
  """Places365 classifier"""

  # -------------------------------------------------------------
  # imports
  
  import os
  from os.path import join
  from pathlib import Path

  import cv2 as cv
  import numpy as np

  from vframe.utils import file_utils, im_utils, logger_utils
  from vframe.models.metadata_item import ClassifyMetadataItem, ClassifyResult
  from vframe.settings.paths import Paths

  # -------------------------------------------------------------
  # init

  log = logger_utils.Logger.getLogger()

  metadata_type = types.Metadata.PLACES365

  # process keyframes
  dir_media = Paths.media_dir(types.Metadata.KEYFRAME, data_store=opt_disk, verified=ctx.opts['verified'])
  opt_size_label = cfg.IMAGE_SIZE_LABELS[opt_size]


  # GoogleNet 365
  dir_places365 = join(cfg.DIR_MODELS,'caffe/places365')
  fp_model = join(dir_places365,'googlenet_places365/googlenet_places365.caffemodel')
  fp_prototxt = join(dir_places365,'googlenet_places365/deploy_googlenet_places365.prototxt')

  # ResNet152 365
  #fp_model = join(dir_places365,'resnet152_places365/resnet152_places365.caffemodel')
  #fp_prototxt = join(dir_places365,'resnet152_places365/deploy_resnet152_places365.prototxt')

  # VGG 365
  #fp_model = join(dir_places365,'vgg16_places365/vgg16_places365.caffemodel')
  #fp_prototxt = join(dir_places365,'vgg16_places365/deploy_vgg16_places365.prototxt')

  # 1365 classes models

  # ResNet152 1365
  #fp_model = join(dir_places365,'resnet152_places365/resnet152_places365.caffemodel')
  #fp_prototxt = join(dir_places365,'resnet152_places365/deploy_resnet152_places365.prototxt')

  # VGG 1365
  #fp_model = join(dir_places365,'resnet152_places365/alexnet_places365.caffemodel')
  #fp_prototxt = join(dir_places365,'alexnet_places365/deploy_alexnet_places365.prototxt')

  # TODO move these to Path/settings file
  # load the classes
  fp_categories = join(dir_places365,'data/categories_places365.txt')
  lines = file_utils.load_text(fp_categories)
  classes = tuple(list(line.strip().split(' ')[0][3:]))

  # initialize dnn with settings for places365
  net = cv.dnn.readNetFromCaffe(fp_prototxt, fp_model)
  im_size_dnn = (224, 224)
  clr_dnn = (104, 117, 123)  # BGR
  dnn_scale = 1
  threshold = 0.8

  # iterate sink
  while True:
    chair_item = yield
    media_record = chair_item.media_record
    sha256 = media_record.sha256
    sha256_tree = file_utils.sha256_tree(sha256)
    dir_sha256 = join(dir_media, sha256_tree, sha256)
    
    # get the keyframe status data to check if images available
    try:
      keyframe_status = media_record.get_metadata(types.Metadata.KEYFRAME_STATUS)
    except Exception as ex:
      log.error('no keyframe metadata. Try: "append -t keyframe_status"')
      return

    # if keyframe images were generated and exist locally
    metadata = {}
    if keyframe_status and keyframe_status.get_status(opt_size):
      try:
        keyframe_metadata = media_record.get_metadata(types.Metadata.KEYFRAME)
      except Exception as ex:
        log.error('no keyframe metadata. Try: "append -t keyframe"')
        return

      # get keyframe indices
      idxs = keyframe_metadata.get_keyframes(opt_density)

      for idx in idxs:
        # get keyframe filepath
        fp_keyframe = join(dir_sha256, file_utils.zpad(idx), opt_size_label, 'index.jpg')
        im = cv.imread(fp_keyframe)
        blob = cv.dnn.blobFromImage(im, dnn_scale, im_size_dnn, clr_dnn)
        net.setInput(blob)
        net_output = net.forward()[0]
        # remove detections below threshold
        top_idxs = np.where(np.array(net_output) > threshold)[0]
        valid_results = [ClassifyResult(x, float(net_output[x])) for x in top_idxs]
        metadata[idx] = valid_results

    # append metadata to chair_item's mapping item
    chair_item.item.set_metadata(metadata_type, ClassifyMetadataItem(metadata))
    
    # send back to generator
    sink.send(chair_item)