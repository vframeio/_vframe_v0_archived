"""
Generate metadata using OpenCV's DNN module
- under development
- about 10FPS? on i7 CPU 12x
- using Python Yolo is much faster w/GPU
"""

import click

from vframe.settings import types
from vframe.utils import click_utils
from vframe.settings import vframe_cfg as cfg

from cli_vframe import processor

@click.command('gen_darknet_coco')
@click.option('-d', '--disk', 'opt_disk',
  default=click_utils.get_default(types.DataStore.HDD),
  type=cfg.DataStoreVar,
  show_default=True,
  help=click_utils.show_help(types.DataStore))
@click.option('-t', '--net-type', 'opt_net',
  type=cfg.DetectorNetVar,
  default=click_utils.get_default(types.DetectorNet.SUBMUNITION),
  help=click_utils.show_help(types.DetectorNet))
@click.option('-g', '--gpu', 'opt_gpu', default=0,
  help='GPU index')
@processor
@click.pass_context
def cli(ctx, sink, opt_disk, opt_net, opt_gpu):
  """Generates detections with CV DNN"""

  # ----------------------------------------------------------------
  # imports

  import os
  from os.path import join
  from pathlib import Path

  import click
  import cv2 as cv
  import numpy as np

  from vframe.utils import click_utils, file_utils, im_utils, logger_utils, dnn_utils
  from vframe.models.metadata_item import DetectMetadataItem, DetectResult
  from vframe.settings.paths import Paths

  # ----------------------------------------------------------------
  # init

  log = logger_utils.Logger.getLogger()


  # TODO externalize function

  # initialize dnn
  dnn_clr = (0, 0, 0)  # mean color to subtract
  dnn_scale = 1/255  # ?
  nms_threshold = 0.4   #Non-maximum suppression threshold
  dnn_px_range = 1  # pixel value range ?
  dnn_crop = False  # probably crop or force resize

  # Use mulitples of 32: 416, 448, 480, 512, 544, 576, 608, 640, 672, 704
  if opt_net == types.DetectorNet.OPENIMAGES:
    metadata_type = types.Metadata.OPENIMAGES
    dnn_size = (608, 608)
    dnn_threshold = 0.875
  elif  opt_net == types.DetectorNet.COCO:
    metadata_type = types.Metadata.COCO
    dnn_size = (416, 416)
    dnn_threshold = 0.925
  elif  opt_net == types.DetectorNet.COCO_SPP:
    metadata_type = types.Metadata.COCO
    dnn_size = (608, 608)
    dnn_threshold = 0.875
  elif  opt_net == types.DetectorNet.VOC:
    metadata_type = types.Metadata.VOC
    dnn_size = (416, 416)
    dnn_threshold = 0.875
  elif  opt_net == types.DetectorNet.SUBMUNITION:
    metadata_type = types.Metadata.SUBMUNITION
    dnn_size = (608, 608)
    dnn_threshold = 0.90

  # Initialize the parameters
  fp_cfg = Paths.darknet_cfg(opt_net=opt_net, data_store=opt_disk, as_bytes=False)
  fp_weights = Paths.darknet_weights(opt_net=opt_net, data_store=opt_disk, as_bytes=False)
  fp_data = Paths.darknet_data(opt_net=opt_net, data_store=opt_disk, as_bytes=False)
  fp_classes = Paths.darknet_classes(opt_net=opt_net, data_store=opt_disk)
  class_names = file_utils.load_text(fp_classes)
  class_idx_lookup = {label: i for i, label in enumerate(class_names)}

  log.debug('fp_cfg: {}'.format(fp_cfg))
  log.debug('fp_weights: {}'.format(fp_weights))
  log.debug('fp_data: {}'.format(fp_data))
  log.debug('fp_classes: {}'.format(fp_classes))

  net = cv.dnn.readNetFromDarknet(fp_cfg, fp_weights)
  net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
  net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

  # ----------------------------------------------------------------
  # process

  # iterate sink
  while True:
    
    chair_item = yield
    
    metadata = {}
    
    for frame_idx, frame in chair_item.keyframes.items():

      frame = im_utils.resize(frame, width=dnn_size[0], height=dnn_size[1])
      blob = cv.dnn.blobFromImage(frame, dnn_scale, dnn_size, dnn_clr, 
        dnn_px_range, crop=dnn_crop)
      
      # Sets the input to the network
      net.setInput(blob)

      # Runs the forward pass to get output of the output layers
      net_outputs = net.forward(dnn_utils.getOutputsNames(net))
      det_results = dnn_utils.nms_cvdnn(net_outputs, dnn_threshold, nms_threshold)
      
      metadata[frame_idx] = det_results

    # append metadata to chair_item's mapping item
    chair_item.item.set_metadata(metadata_type, DetectMetadataItem(metadata))
  
    # ----------------------------------------------------------------
    # yield back to the processor pipeline

    # send back to generator
    sink.send(chair_item)


