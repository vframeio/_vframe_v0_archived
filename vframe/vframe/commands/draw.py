"""
Add/remove metadata to media records
"""

import click

from vframe.utils import click_utils, draw_utils
from vframe.settings import types
from vframe.settings import vframe_cfg as cfg
from vframe.models.bbox import BBox
from cli_vframe import processor

import matplotlib as mpl
import matplotlib.cm as mplcm
import matplotlib.pyplot as plt
from matplotlib import cm

def get_color_map(cmap='prism', ncolors=20, as_hex=False, reverse=False, bgr=True):
  norm  = mpl.colors.Normalize(vmin=0, vmax=ncolors-1)
  scalars = mplcm.ScalarMappable(norm=norm, cmap=cmap)
  colors = [scalars.to_rgba(i) for i in range(ncolors)]
  colors = [(int(255*c[0]),int(255*c[1]),int(255*c[2])) for c in colors]  
  if reverse:
    colors = colors[::-1]
  if bgr:
    colors = [c[::-1] for c in colors]
  if as_hex:
    colors = ['#{:02x}{:02x}{:02x}'.format(c[0],c[1],c[2]) for c in colors]
  return colors

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
@click.option('--stroke-width', 'opt_stroke_width', default=3,
  help='Rectangle outline stroke width (px')
@click.option('--stroke-color', 'opt_stroke_color', 
  type=(int, int, int), default=(0,255,0),
  help='Rectangle color')
@click.option('--text-color', 'opt_text_color', 
  type=(int, int, int), default=(0,0,0),
  help='Text color')
@processor
@click.pass_context
def cli(ctx, sink, opt_metadata, opt_disk, opt_stroke_width, opt_stroke_color, opt_text_color):
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
    # TODO add class file
  elif opt_metadata == types.Metadata.TEXT_ROI:
    pass
  elif opt_metadata == types.Metadata.FACE_ROI:
    pass
  
  # get colors for stroke
  colors = get_color_map(cmap='autumn', reverse=True, ncolors=len(classes))

  # TODO externalize function
  
  # -------------------------------------------------
  # process 
  
  while True:
    
    chair_item = yield

    drawframes = {}  # new drawframes

    # ---------------------------------------------------------------
    # draw on images, assume detection results (not classify)

    detection_metadata = chair_item.get_metadata(opt_metadata)

    for frame_idx in chair_item.drawframes.keys():

      drawframe = chair_item.drawframes.get(frame_idx)
      imh, imw = drawframe.shape[:2]

      detection_results = detection_metadata.metadata.get(frame_idx)
      
      for detection_result in detection_results:

        if opt_metadata == types.Metadata.COCO \
          or opt_metadata == types.Metadata.SUBMUNITION \
          or opt_metadata == types.Metadata.VOC \
          or opt_metadata == types.Metadata.OPENIMAGES:
          # draw object detection boxes and labels
          log.debug(detection_result)
          frame = draw_utils.draw_detection_result(drawframe, classes, detection_result, 
            imw, imh, stroke_weight=opt_stroke_width, 
            rect_color=colors[detection_result.idx], text_color=opt_text_color)

        elif opt_metadata == types.Metadata.TEXT_ROI:
          frame = draw_utils.draw_roi(drawframe, detection_result, 
            imw, imh, text='TEXT', stroke_weight=opt_stroke_width, 
            rect_color=opt_stroke_color, text_color=opt_text_color)
        elif opt_metadata == types.Metadata.FACE_ROI:
          frame = draw_utils.draw_roi(drawframe, detection_result, 
            imw, imh, text='FACE', stroke_weight=opt_stroke_width, 
            rect_color=opt_stroke_color, text_color=opt_text_color)
        
      # add to current items drawframes dict
      drawframes[frame_idx] = drawframe

    chair_item.set_drawframes(drawframes)


    # ------------------------------------------------
    # rebuild the generator
    sink.send(chair_item)
