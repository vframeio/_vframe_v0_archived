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
@click.option('-a', '--action', 'opt_action', required=True,
  type=cfg.ActionVar,
  default=click_utils.get_default(types.Action.ADD),
  help=click_utils.show_help(types.Action))
@click.option('-t', '--net-type', 'opt_net',
  type=cfg.DetectorNetVar,
  default=click_utils.get_default(types.DetectorNet.COCO),
  help=click_utils.show_help(types.DetectorNet))
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
def cli(ctx, sink, opt_action, opt_net, opt_disk, opt_stroke_weight, opt_stroke_color, opt_text_color):
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
  

  fp_classes = Paths.darknet_classes(data_store=opt_disk, opt_net=opt_net)
  classes = file_utils.load_text(fp_classes)  # returns list in idx order
  
  # TODO externalize function
  # convert net type into metadata type
  if opt_net == types.DetectorNet.COCO:
    opt_metadata = types.Metadata.COCO
  elif opt_net == types.DetectorNet.COCO_SPP:
    opt_metadata = types.Metadata.COCO  
  elif opt_net == types.DetectorNet.VOC:
    opt_metadata = types.Metadata.VOC  
  elif opt_net == types.DetectorNet.OPENIMAGES:
    opt_metadata = types.Metadata.OPENIMAGES  
  elif opt_net == types.DetectorNet.SUBMUNITION:
    opt_metadata = types.Metadata.SUBMUNITION
  else:
    log.error('{} not a valid type'.format(opt_net))
    return

  # -------------------------------------------------
  # process 
  
  while True:
    
    chair_item = yield

    drawframes = {}

    # ensure frames and metadata exist
    # if not chair_item.get_metadata(opt_metadata):

    if opt_action == types.Action.ADD:

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

          frame = draw_utils.draw_detection_result(drawframe, classes, detection_result, imw, imh, 
            stroke_weight=opt_stroke_weight, rect_color=opt_stroke_color, text_color=opt_text_color)
          
        # add to current items drawframes dict
        drawframes[frame_idx] = drawframe

      # after drawing all, append imges to chair_item
      # reset drawframes ?
      chair_item.set_drawframes(drawframes)

    elif opt_action == types.Action.RM:
      
      # ---------------------------------------------------------------
      # remvoe image data to free RAM

      chair_item.remove_drawframes()

    # ------------------------------------------------
    # rebuild the generator
    sink.send(chair_item)
