"""
Add/remove metadata to media records
"""

import click

from vframe.utils import click_utils
from vframe.settings import types
from vframe.settings import vframe_cfg as cfg

from cli_vframe import processor


# --------------------------------------------------------
# Displays images, no video yet
# --------------------------------------------------------
@click.command()
@click.option('--delay', 'opt_delay', default=150,
  type=click.IntRange(1, 1000),
  show_default=True,
  help='Delay between images in millis. 1 is fastest.')
@click.option('--dispose/--no-dispose', 'opt_dispose', is_flag=True, default=True,
  help='Dispose image after display to free RAM')
@processor
@click.pass_context
def cli(ctx, sink, opt_delay, opt_dispose):
  """Displays images"""
  
  # -------------------------------------------------
  # imports 

  import os
  
  import cv2 as cv
  import numpy as np
  from PIL import Image
  from tqdm import tqdm

  from vframe.settings.paths import Paths
  from vframe.utils import file_utils, logger_utils, chair_utils
  
  
  # -------------------------------------------------
  # init 

  log = logger_utils.Logger.getLogger()
  log.info('opt_delay: {}'.format(opt_delay))
  
  # -------------------------------------------------
  # process 
  
  while True:
    
    chair_item = yield

    for frame_idx, frame in chair_item.drawframes.items():
      cv.imshow('vframe', frame)
      chair_utils.handle_keyboard(ctx, opt_delay)

    # ------------------------------------------------
    # rebuild the generator
    sink.send(chair_item)

  # this makes it wait forever
  cv2.waitKey(0)
