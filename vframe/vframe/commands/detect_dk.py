"""
Generates metadata using Yolo/Darknet Python interface
- about 20-30 FPS on NVIDIA 1080 Ti GPU
- SPP currently not working
- enusre image size matches network image size

"""
import click

from vframe.settings import types
from vframe.utils import click_utils
from vframe.settings import vframe_cfg as cfg

from cli_vframe import processor


@click.command()
@click.option('-d', '--disk', 'opt_disk',
  default=click_utils.get_default(types.DataStore.SSD),
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
  """Generates detections with Darknet"""

  # -------------------------------------------------
  # imports 

  import os
  from os.path import join
  from pathlib import Path
  from io import BytesIO

  import PIL.Image
  import numpy as np
  import requests
  import cv2 as cv
  import numpy as np
  # temporary fix, update pydarknet to >= 0.1rc12
  # pip install yolo34py-gpu==0.1rc13
  os.environ['CUDA_VISIBLE_DEVICES'] = str(opt_gpu)
  import pydarknet
  from pydarknet import Detector
  from pydarknet import Image as DarknetImage

  from vframe.utils import file_utils, im_utils, logger_utils
  from vframe.models.metadata_item import DetectMetadataItem, DetectResult
  from vframe.settings.paths import Paths


  
  # -------------------------------------------------
  # initialize

  log = logger_utils.Logger.getLogger()

  # Initialize the parameters
  fp_cfg = Paths.darknet_cfg(opt_net=opt_net, data_store=opt_disk)
  fp_weights = Paths.darknet_weights(opt_net=opt_net, data_store=opt_disk)
  fp_data = Paths.darknet_data(opt_net=opt_net, data_store=opt_disk)
  fp_classes = Paths.darknet_classes(opt_net=opt_net, data_store=opt_disk)
  log.debug('fp_classes: {}'.format(fp_classes))
  class_names = file_utils.load_text(fp_classes)
  class_idx_lookup = {label: i for i, label in enumerate(class_names)}

  # init Darknet detector
  # pydarknet.set_cuda_device(opt_gpu)  # not yet implemented in 0.1rc12
  net = Detector(fp_cfg, fp_weights, 0, fp_data)

  # # network input w, h
  nms_thresh = 0.45
  hier_thresh = 0.5
  conf_thresh = 0.5


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

    # TODO: function to collapse hierarchical detections into parent class
    # flatten hierarchical objects
    # ao25_idxs = [0, 1, 2, 3, 7, 8, 9, 11]
    # ao25_idx = 0
    # shoab_idxs = [6, 10]
    # shoab_idx = 12
    # cassette_idxs = [4, 5]
    # cassette_idx = 5

    # for k, v in class_idx_lookup.copy().items():
    #   if v in ao25_idxs:
    #     v = ao25_idx
    #   elif v in shoab_idxs:
    #     v = shoab_idx
    #   elif v in cassette_idxs:
    #     v = cassette_idx
    #   class_idx_lookup[k] = v  


  # -------------------------------------------------
  # process 
  
  while True:

    chair_item = yield
    
    metadata = {}

    for frame_idx, frame in chair_item.keyframes.items():
    
      # -------------------------------------------
      # Start DNN
      
      frame = im_utils.resize(frame, width=dnn_size[0], height=dnn_size[1])
      imh, imw = frame.shape[:2]
      frame_dk = DarknetImage(frame)
      net_outputs = net.detect(frame_dk, thresh=conf_thresh, hier_thresh=hier_thresh, nms=nms_thresh)
      # threshold
      net_outputs = [x for x in  net_outputs if float(x[1]) > dnn_threshold]

      # append as metadata
      det_results = []
      for cat, score, bounds in net_outputs:
        cx, cy, w, h = bounds
        # TODO convert to BBox()
        x1, y1 = ( int(max(cx - w / 2, 0)), int(max(cy - h / 2, 0)) )
        x2, y2 = ( int(min(cx + w / 2, imw)), int(min(cy + h / 2, imh)) )
        class_idx = class_idx_lookup[cat.decode("utf-8")]
        rect_norm = (x1/imw, y1/imh, x2/imw, y2/imh)
        det_results.append( DetectResult(class_idx, float(score), rect_norm) )
        
      # display to screen
      # TODO: replace this with drawing functions
      
      metadata[frame_idx] = det_results
    
    # append metadata to chair_item's mapping item
    #log.debug('metadata: {}'.format(metadata))

    chair_item.set_metadata(metadata_type, DetectMetadataItem(metadata))
  

    # send back to generator
    sink.send(chair_item)
