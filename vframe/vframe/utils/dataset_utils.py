import os
from os.path import join
import logging

from vframe.settings import types
from vframe.settings import vframe_cfg as cfg


def filter_quality(items, quality_min, quality_max=None):


  """Filters list of video input items based on VideoQuality scores"""
  log = logging.getLogger()
  log.info('filtering min: {} to max: {}'.format(quality_min.name.lower()))

  items_copy = items.copy()
  items = {}
  items_failed = {}  # for quality assurance, print num failed
  threshold_min = cfg.VIDEO_QUALITY[quality_min]
  if quality_max:
    threshold_max = cfg.VIDEO_QUALITY[quality_max]
  else:
    threshold_max = None


  for sha256, item in items_copy.items():
    
    try:
      mediainfo = item.metadata.get(types.Metadata.MEDIAINFO)
    except Exception as ex:
      log.error('no metadata: '.format(sha256))
      return items

    try:
      video_data = mediainfo.metadata.get(types.MediainfoMetadata.VIDEO)
    except Exception as ex:
      log.error('no video data: {}'.format(sha256))
      return items

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
            items[sha256] = item
        else:
          # min criteria met
          items[sha256] = item
    else:
      #log.info('quality fail w: {}, h: {}, frame_rate: {}, frame_height: {}'.format(w, h, frame_rate, frame_count))
      items_failed[sha256] = item

  log.info('passed: {:,}, failed: {:,}'.format(len(items), len(items_failed)))
  return items
