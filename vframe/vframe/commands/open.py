"""
Source is a generator and is used at the beginning of the processor workflows
"""
from os.path import join
from pathlib import Path
import click

from vframe.utils import click_utils
from vframe.settings import types
from vframe.settings import vframe_cfg as cfg

from cli_vframe import generator

# --------------------------------------------------------
# Add source items to chain
# Sources can be original mappings or metadata, which always include mappings
# --------------------------------------------------------

@click.command()
@click.option('-i', '--input', 'fp_in', required=True, default=FP_IN_F,
  help='Override file input path')
@generator
@click.pass_context
def cli(ctx, sink, fp_in):
  """Add mappings data to chain"""
  
  # -------------------------------------------------------------
  # imports

  import os
  import logging

  import cv2 as cv
  from tqdm import tqdm

  from vframe.settings.paths import Paths
  from vframe.utils import file_utils, logger_utils
  from vframe.models.chair_item import ChairItem, VideoChairItem, PhotoChairItem, VideoKeyframeChairItem
  from vframe.utils import logger_utils
  
  
  # ---------------------------------------------------------------
  # init

  log = logger_utils.Logger.getLogger()
  log.info('opt_format: {}'.format(fp_in))

  # assume video
  ext = file_utils.get_ext(fp_in)
  media_format = file_utils.ext_media_format(ext)  # enum type
  
  # check file existence
  if not Path(fp_in).exists():
    log.error('file not found: {}'.format(fp_in))
    return

  if media_format == types.MediaFormat.PHOTO:
    log.debug('photo type')
    sink.send( PhotoChairItem(ctx, fp_in) )

  elif media_format == types.MediaFormat.VIDEO:
    log.debug('video type')
    from imutils.video import FileVideoStream
    fvs = FileVideoStream(fp_in, transform=None).start()
    import time
    time.sleep(1.0)
    stream = fvs.stream
    ctx.opts['last_display_ms'] = time.time()
    ctx.opts['fps'] = stream.get(cv.CAP_PROP_FPS)
    ctx.opts['mspf'] = int(1 / ctx.opts['fps'] * 1000)  # milliseconds per frame

    # loop over frames from the video file stream
    frame_idx = 0
    while fvs.running():
      frame = fvs.read()
      sink.send( VideoKeyframeChairItem(ctx, frame, frame_idx) )
      frame_idx += 1      
  

 
