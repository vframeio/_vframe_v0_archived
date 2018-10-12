"""
Add/remove metadata to media records
"""

import click

from vframe.utils import click_utils, draw_utils
from vframe.settings import types
from vframe.settings import vframe_cfg as cfg
from vframe.models.bbox import BBox
from cli_vframe import processor


# --------------------------------------------------------
# Displays images, no video yet
# --------------------------------------------------------
@click.command()
@click.option('-t', '--metadata-type', 'opt_metadata',
  type=cfg.MetadataVar,
  default=click_utils.get_default(types.Metadata.COCO),
  help=click_utils.show_help(types.Metadata))
@click.option('-d', '--disk', 'opt_disk',
  default=click_utils.get_default(types.DataStore.HDD),
  type=cfg.DataStoreVar,
  show_default=True,
  help=click_utils.show_help(types.DataStore))
@click.option('--stroke-weight', 'opt_stroke_weight', default=2,
  help='Rectangle outline stroke weight')
@click.option('--stroke-color', 'opt_stroke_color', 
  type=(int, int, int), default=(0,255,0),
  help='Rectangle color')
@click.option('--text-color', 'opt_text_color', 
  type=(int, int, int), default=(0,0,0),
  help='Text color')
@processor
@click.pass_context
def cli(ctx, sink, opt_metadata, opt_disk, opt_stroke_weight, opt_stroke_color, opt_text_color):
  """Displays images"""
  
  # -------------------------------------------------
  # imports 

  import os
  
  import cv2 as cv
  import numpy as np

  from vframe.settings.paths import Paths
  from vframe.utils import file_utils, logger_utils, chair_utils
  
  
  # -------------------------------------------------
  # init 

  log = logger_utils.Logger.getLogger()
  
  # load class labels
  if opt_metadata == types.Metadata.COCO:
    opt_net = types.DetectorNet.COCO
    fp_classes = Paths.darknet_classes(data_store=opt_disk, opt_net=opt_net)
    classes = file_utils.load_text(fp_classes)  # returns list in idx order
  if opt_metadata == types.Metadata.OPENIMAGES:
    opt_net = types.DetectorNet.OPENIMAGES
    fp_classes = Paths.darknet_classes(data_store=opt_disk, opt_net=opt_net)
    classes = file_utils.load_text(fp_classes)  # returns list in idx order
  elif opt_metadata == types.Metadata.SUBMUNITION:
    opt_net = types.DetectorNet.SUBMUNITION
    fp_classes = Paths.darknet_classes(data_store=opt_disk, opt_net=opt_net)
    classes = file_utils.load_text(fp_classes)  # returns list in idx order
  elif opt_metadata == types.Metadata.PLACES365:
    opt_net = types.ClassifyNet.PLACES365
    pass
  elif opt_metadata == types.Metadata.TEXTROI:
    pass
  
  # TODO externalize function
  
  # -------------------------------------------------
  # process 
  
  while True:
    
    chair_item = yield

    drawframes = {}  # new drawframes

    # ---------------------------------------------------------------
    # draw on images, assume detection results (not classify)

    detection_metadata = chair_item.get_metadata(opt_metadata)
    log.debug('frames: {}'.format(detection_metadata))

    for frame_idx in chair_item.drawframes.keys():

      drawframe = chair_item.drawframes.get(frame_idx)
      imh, imw = drawframe.shape[:2]

      detection_results = detection_metadata.metadata.get(frame_idx)

      log.debug('detection_results: {}'.format(detection_results))
      
      for detection_result in detection_results:

        if opt_metadata == types.Metadata.COCO \
          or opt_metadata == types.Metadata.SUBMUNITION \
          or opt_metadata == types.Metadata.VOC \
          or opt_metadata == types.Metadata.OPENIMAGES:
          # draw object detection boxes and labels
          frame = draw_utils.draw_detection_result(drawframe, classes, detection_result, 
            imw, imh, stroke_weight=opt_stroke_weight, 
            rect_color=opt_stroke_color, text_color=opt_text_color)

        elif opt_metadata == types.Metadata.TEXTROI:
          frame = draw_utils.draw_scenetext_result(drawframe, detection_result, 
            imw, imh, stroke_weight=opt_stroke_weight, 
            rect_color=opt_stroke_color, text_color=opt_text_color)
        
      # add to current items drawframes dict
      drawframes[frame_idx] = drawframe

    chair_item.set_drawframes(drawframes)


    # ------------------------------------------------
    # rebuild the generator
    sink.send(chair_item)
