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
# FP_IN_DEFAULT = '/data_store_hdd/apps/syrianarchive/media/test_videos/0a0b4feb75ccec99e44a44120734d75ab1a235c274d0577a191d631297cccbc1.mp4'
FP_IN_NAS = '/data_store_nas/datasets/syrianarchive/media/videos/'
FP_IN_DIR = '/data_store_hdd/apps/syrianarchive/media/demo_videos/'
FP_IN_A = join(FP_IN_DIR, '29bffdf620ccbf38cc8481aa73eb23fd5ab821697114109d63e57c7024bbedf8.mp4')
FP_IN_B = join(FP_IN_DIR,'a3ea6c018a81a4e9c77903c1ca92e68e5133f38e1a0903957f8b20d0a6a03fa8.mp4')
FP_IN_C = join(FP_IN_DIR, 'f1ba90e063f12f3dd22f51a78920c1baadd020b0f61fc0d4b3d07c06e0456b5f.mp4')
FP_IN_D = join(FP_IN_DIR, '7545a37e8d596fdcc5e58b6d49a4d3703143737f5d7702ec3f5cac847902c50e.mp4')
FP_IN_E = join(FP_IN_NAS, '1c7/02c/23f/1c702c23f4a7388b94adcc76c27a94b5cdbb652a32b920417b6e7bbd58fa8f77.mp4')
FP_IN_F = join(FP_IN_NAS, '6c7/88e/ae2/6c788eae284a7894827a567eae7ddb9562aed8e2c38eaeb3aa66e11bc6812525.mp4')

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
  

 
