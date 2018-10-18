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
@click.option('-i', '--input', 'fp_in', required=True,
  help='Override file input path')
@click.option('--size', 'opt_size_type',
  type=cfg.ImageSizeVar,
  default=click_utils.get_default(types.ImageSize.MEDIUM),
  help=click_utils.show_help(types.ImageSize))  # TODO move to add_images
@generator
@click.pass_context
def cli(ctx, sink, fp_in, opt_size_type):
  """Add mappings data to chain"""
  
  # -------------------------------------------------------------
  # imports

  import os
  import logging

  import cv2 as cv
  from tqdm import tqdm
  import imutils

  from vframe.settings.paths import Paths
  from vframe.utils import file_utils, logger_utils
  from vframe.models.chair_item import ChairItem, PhotoChairItem, VideoKeyframeChairItem
  from vframe.utils import logger_utils, im_utils
  
  
  # ---------------------------------------------------------------
  # init

  log = logger_utils.Logger.getLogger()
  log.info('fp_in: {}'.format(fp_in))

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
    
    import time
    from imutils.video import FileVideoStream

    opt_size = cfg.IMAGE_SIZES[opt_size_type]

    # TODO this should be in "add_images" command
    def frame_transform(frame):
      return im_utils.resize(frame, width=opt_size)

    fvs = FileVideoStream(fp_in, transform=frame_transform).start()
    time.sleep(1.0)  # recommended delay
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
  

 
