"""
Using CV DNN for classification
"""
import click

from vframe.settings import types
from vframe.utils import click_utils
from vframe.settings import vframe_cfg as cfg

from cli_vframe import processor


@click.command()
@click.option('-t', '--net-type', 'opt_net',
  default=click_utils.get_default(types.ClassifyNet.PLACES365),
  type=cfg.ClassifyNetVar,
  help=click_utils.show_help(types.ClassifyNet))
@click.option('-d', '--disk', 'opt_disk',
  default=click_utils.get_default(types.DataStore.SSD),
  type=cfg.DataStoreVar,
  show_default=True,
  help=click_utils.show_help(types.DataStore))
@processor
@click.pass_context
def cli(ctx, sink, opt_net, opt_disk):
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

  # -------------------------------------------------------------
  # init

  log = logger_utils.Logger.getLogger()


  # TODO externalize config
  if opt_net == types.ClassifyNet.PLACES365:
    # GoogleNet 365
    metadata_type = types.Metadata.PLACES365
    dir_places365 = join(cfg.DIR_MODELS_VFRAME,'caffe/places365')
    dir_googlenet = join(dir_places365,'googlenet_places365')
    fp_model = join(dir_googlenet, 'googlenet_places365.caffemodel')
    fp_prototxt = join(dir_googlenet,'deploy_googlenet_places365.prototxt')
    fp_classes = join(dir_places365,'data/categories_places365.txt')
    
    lines = file_utils.load_text(fp_classes)
    classes = [line.strip().split(' ')[0][3:] for line in lines]

    im_size_dnn = (224, 224)
    clr_dnn = (104, 117, 123)  # BGR
    dnn_scale = 1
    threshold = 0.8
  else:
    log.error('{} not yet implemented'.format(opt_net))
    return

  # initialize dnn with settings for places365
  net = cv.dnn.readNetFromCaffe(fp_prototxt, fp_model)

  # iterate sink
  while True:
  
    chair_item = yield

    metadata = {}

    # check if no images
    if not len(chair_item.keyframes.keys()) > 0:
      log.warn('no images for {}'.format(chair_item.sha256))  #  try adding "images" to command?

    # iterate keyframes and extract feature vectors as serialized data
    for frame_idx, frame in chair_item.keyframes.items():
      blob = cv.dnn.blobFromImage(frame, dnn_scale, im_size_dnn, clr_dnn)
      net.setInput(blob)
      net_output = net.forward()[0]
      # remove detections below threshold
      top_idxs = np.where(np.array(net_output) > threshold)[0]
      valid_results = [ClassifyResult(x, float(net_output[x])) for x in top_idxs]
      metadata[frame_idx] = valid_results

    # append metadata to chair_item's mapping item
    chair_item.item.set_metadata(metadata_type, ClassifyMetadataItem(metadata))
    
    # send back to generator
    sink.send(chair_item)