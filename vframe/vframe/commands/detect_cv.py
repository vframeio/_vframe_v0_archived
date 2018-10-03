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
@click.option('--display/--no-display', 'opt_display', is_flag=True,
  help='Display the image')
@processor
@click.pass_context
def cli(ctx, sink, opt_disk, opt_density, opt_size, opt_display):
  """Generates detections with CV DNN"""

  # ----------------------------------------------------------------
  # imports

  import os
  from os.path import join
  from pathlib import Path

  import click
  import cv2 as cv
  import numpy as np

  from vframe.utils import click_utils, file_utils, im_utils, logger_utils
  from vframe.models.metadata_item import DetectMetadataItem, DetectResult
  from vframe.settings.paths import Paths

  # ----------------------------------------------------------------
  # init

  log = logger_utils.Logger.getLogger()

  metadata_type = types.Metadata.COCO

  # directory for images
  dir_media = Paths.media_dir(types.Metadata.KEYFRAME, data_store=opt_disk, verified=ctx.opts['verified'])
  opt_size_label = cfg.IMAGE_SIZE_LABELS[opt_size]

  # Initialize the parameters


  # TODO externalize function

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

  # TODO externalize to Paths    
  DIR_DARKNET = join(cfg.DIR_MODELS, 'darknet/pjreddie')
  fp_weights = join(DIR_DARKNET, 'weights/yolov3.weights')
  fp_cfg = join(DIR_DARKNET, 'cfg/yolov3.cfg')
  fp_classes = join(DIR_DARKNET, 'data/coco.names')

  classes = file_utils.load_text(fp_classes)
  net = cv.dnn.readNetFromDarknet(fp_cfg, fp_weights)
  net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
  net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

  # TODO sort this out

  # # network input w, h
  nms_thresh = 0.45
  hier_thresh = 0.5
  conf_thresh = 0.5

  # initialize dnn
  dnn_size = (416, 416)  # network input w, h
  dnn_clr = (0, 0, 0)  # mean color to subtract
  dnn_scale = 1/255
  dnn_threshold = 0.85  #Confidence threshold
  nms_threshold = 0.4   #Non-maximum suppression threshold
  dnn_px_range = 1  # pixel value range
  dnn_crop = False

  # ----------------------------------------------------------------
  # process

  # iterate sink
  while True:
    
    chair_item = yield
    
    metadata = {}
    
    for frame_idx, frame in chair_item.keyframes.items():

      blob = cv.dnn.blobFromImage(frame, dnn_scale, dnn_size, dnn_clr, 
        dnn_px_range, crop=dnn_crop)
      
      # Sets the input to the network
      net.setInput(blob)

      # Runs the forward pass to get output of the output layers
      net_outputs = net.forward(dnn_utils.getOutputsNames(net))

      det_results = dnn_utils.post_process(im, net_outputs, dnn_threshold, nms_threshold)
      
      metadata[frame_idx] = det_results

    # append metadata to chair_item's mapping item
    chair_item.item.set_metadata(metadata_type, DetectMetadataItem(metadata))
  
    # ----------------------------------------------------------------
    # yield back to the processor pipeline

    # send back to generator
    sink.send(chair_item)


