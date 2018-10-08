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
@processor
@click.pass_context
def cli(ctx, sink, opt_delay):
  """Displays images"""
  
  # -------------------------------------------------
  # imports 

  import os
  from time import time

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
  
  ctx.opts.setdefault('last_frame_ms', time())
  log.debug('last_frame_ms: {}'.format(ctx.opts['last_frame_ms']))

  video_buffer_adj = -5  # milliseconds to adjust delay

  # -------------------------------------------------
  # process 
  
  while True:
    
    chair_item = yield

    if chair_item.chair_type == types.ChairItemType.MEDIA_RECORD:
      log.debug('display chair media record')
      for frame_idx, frame in chair_item.drawframes.items():
        cv.imshow('vframe', frame)
        log.debug('show frame')
        chair_utils.handle_keyboard(ctx, opt_delay)

    elif chair_item.chair_type == types.ChairItemType.PHOTO:
      pass
    elif chair_item.chair_type == types.ChairItemType.VIDEO_KEYFRAME:
      frame = chair_item.drawframe
      cv.imshow('vframe', frame)
      # handle keyboard, autocalculate threshold frame rate
      ms_elapsed = time() - ctx.opts['last_display_ms']
      delta_ms = int(max(ctx.opts['mspf'] - ms_elapsed + video_buffer_adj, 1))
      chair_utils.handle_keyboard(ctx, delta_ms)  # override delay
      ctx.opts['last_display_ms'] = time()

    elif chair_item.chair_type == types.ChairItemType.VIDEO:
      log.debug('video has: {} frames'.format(len(chair_item.drawframes.items())))
      for frame_idx, frame in chair_item.drawframes.items():
        cv.imshow('vframe', frame)
        # handle keyboard, autocalculate threshold frame rate
        ms_elapsed = time() - chair_item.last_display_ms
        delta_ms = int(max(chair_item.mspf - ms_elapsed + video_buffer_adj, 1))
        chair_utils.handle_keyboard(ctx, delta_ms)  # override delay
        chair_item.last_display_ms = time()


    # ------------------------------------------------
    # rebuild the generator
    sink.send(chair_item)

  # this makes it wait forever
  cv2.waitKey(0)
