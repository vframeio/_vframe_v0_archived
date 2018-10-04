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
@click.option('-o', '--output', 'opt_dir_media', required=True,
  help='Path to media folder')
@processor
@click.pass_context
def cli(ctx, sink, opt_dir_media):
  """Saves keyframes for still-frame-video"""

  
  # -------------------------------------------------
  # imports 

  from os.path import join

  import cv2 as cv
  
  from vframe.utils import file_utils, logger_utils
  from vframe.settings.paths import Paths

  
  # -------------------------------------------------
  # initialize

  log = logger_utils.Logger.getLogger()
  log.debug('init saves images')
  file_utils.mkdirs(opt_dir_media)
  
  
  # -------------------------------------------------
  # process 
  
  frame_count = 0
  while True:
    
    chair_item = yield
    
    for frame_idx, frame in chair_item.keyframes.items():
      # save frame to the output folder
      fp_im = join(opt_dir_media, 'frame_{}.png'.format(file_utils.zpad(frame_count)))
      cv.imwrite(fp_im, frame) 
      frame_count += 1

    # ------------------------------------------------------------
    # send back to generator

    sink.send(chair_item)
