"""
Add/remove metadata to media records
"""

import click

from vframe.utils import click_utils
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
  default=click_utils.get_default(types.DataStore.SSD),
  type=cfg.DataStoreVar,
  show_default=True,
  help=click_utils.show_help(types.DataStore))
@click.option('-c', '--color', 'opt_color')
@processor
@click.pass_context
def cli(ctx, sink, opt_action, opt_net, opt_disk, opt_color):
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
  log.info('opt_color: {}'.format(opt_color))

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

  font = cv.FONT_HERSHEY_SIMPLEX
  tx_offset = 4
  ty_offset = 5
  tx2_offset = 2 * tx_offset
  ty2_offset = 2 * ty_offset
  tx_scale = 0.4
  tx_clr = (0,0,0)
  rect_clr = (0,255,0)
  stroke_weight = 2
  tx_weight = 1

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
          bbox = BBox.from_norm_coords(detection_result.rect, imw, imh)
          class_idx = detection_result.idx
          score = detection_result.score
          
          # draw border
          pt1, pt2 = bbox.pt1, bbox.pt2
          cv.rectangle(drawframe, pt1.tuple() , pt2.tuple(), rect_clr, thickness=stroke_weight)

          # prepare label
          label = '{} ({:.2f})'.format(classes[class_idx].upper(), float(score))
          log.debug('label: {}'.format(label))
          tw, th = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, tx_scale, tx_weight)[0]
          
          # draw label bg
          rect_pt2 = (pt1.x + tw + tx2_offset, pt1.y + th + ty2_offset)
          cv.rectangle(drawframe, pt1.tuple(), rect_pt2, (0, 255, 0), -1)
          # draw label
          cv.putText(drawframe, label, pt1.offset(tx_offset, 3*ty_offset), font, tx_scale, tx_clr, tx_weight)

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

  # this makes it wait forever
  cv2.waitKey(0)
