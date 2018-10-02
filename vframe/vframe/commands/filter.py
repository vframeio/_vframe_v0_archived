"""
Filter media records by video size
"""

import click

from vframe.utils import click_utils
from vframe.settings import types
from vframe.settings import vframe_cfg as cfg

from cli_vframe import processor


# --------------------------------------------------------
# Add source items to chain
# --------------------------------------------------------
@click.command('quality', short_help='Filters mapping and metadata')
@click.option('--min', 'opt_quality_min',
  default=click_utils.get_default(types.VideoQuality.HD),
  type=cfg.VideoQualityVar,
  show_default=True,
  help=click_utils.show_help(types.VideoQuality))
@click.option('--max', 'opt_quality_max',
  default=None,
  type=cfg.VideoQualityVar,
  show_default=True,
  help=click_utils.show_help(types.VideoQuality))
@processor
@click.pass_context
def cli(ctx, sink, opt_quality_min, opt_quality_max):
  """Filters video quality by frame size, count, and rate
  Videos down to and including the min VideoQualityVar are passed through
  Videos up to and including the max VideoQuality are passed through"""


  # -------------------------------------------------
  # imports

  import os

  from tqdm import tqdm

  from vframe.settings import vframe_cfg as cfg
  from vframe.settings.paths import Paths
  from vframe.utils import file_utils, logger_utils


  # -------------------------------------------------
  # process

  log = logger_utils.Logger.getLogger()

  log.info('filter from {} to {}'.format(opt_quality_min, opt_quality_max))

  # accumulate items
  chair_items = []
  while True:
    try:
      chair_items.append( (yield) )
    except GeneratorExit as ex:
      break

  # swap arrays for filtering
  chair_items_copy = chair_items.copy()
  chair_items = []
  chair_items_failed = []  # for quality assurance, print num failed

  # get video quality threshold objects
  threshold_min = cfg.VIDEO_QUALITY[opt_quality_min]
  if opt_quality_max:
    threshold_max = cfg.VIDEO_QUALITY[opt_quality_max]
  else:
    threshold_max = None

  # iterate items and append valid items
  for chair_item in chair_items_copy:

    sha256 = chair_item.sha256
    item = chair_item.item

    try:
      mediainfo = item.get_metadata(types.Metadata.MEDIAINFO)
    except Exception as ex:
      log.error('"mediainfo" is required in the metadata fields')
      return

    try:
      video_data = mediainfo.metadata.get(types.MediainfoMetadata.VIDEO)
    except Exception as ex:
      log.error('"mediainfo" is required in the metadata fields')
      log.error('Error: try "append -t mediainfo"')
      return chair_items

    codec = video_data.get('codec_id', '')
    w = video_data.get('width', 0)
    h = video_data.get('height', 0)
    frame_count = video_data.get('frame_count', 0)
    frame_rate =  video_data.get('frame_rate', 0)

    # threshold min
    if ((w >= threshold_min.width and h >= threshold_min.height) \
      or (h >= threshold_min.width and w >= threshold_min.height)) \
      and frame_count >= threshold_min.frame_count \
      and frame_rate >= threshold_min.frame_rate \
      and codec == threshold_min.codec:
        # filter ceiling if max is not none
        if threshold_max:
          if ((w <= threshold_max.width and h <= threshold_max.height) \
            or (h <= threshold_max.width and w <= threshold_max.height)):
            # min and max criteria met
            chair_items.append(chair_item)
        else:
          # min criteria met
          chair_items.append(chair_item)
    else:
      #log.info('quality fail w: {}, h: {}, frame_rate: {}, frame_height: {}'.format(w, h, frame_rate, frame_count))
      chair_items_failed.append(chair_item)

  log.info('passed: {:,}, failed: {:,}'.format(len(chair_items), len(chair_items_failed)))
  
  # update items
  ctx.opts['num_items'] = len(chair_items)


  # -------------------------------------------------
  # rebuild the generator

  for item in tqdm(chair_items):
      sink.send(item)