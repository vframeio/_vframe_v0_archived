"""Generates keyframe image stauts, whether file exists locally
"""
import click

from vframe.settings import types
from vframe.utils import click_utils
from vframe.settings import vframe_cfg as cfg

from cli_vframe import processor

@click.command('keyframe_status')
@click.option('-d', '--disk', 'opt_disk',
  default=click_utils.get_default(types.DataStore.SSD),
  type=cfg.DataStoreVar,
  show_default=True,
  help=click_utils.show_help(types.DataStore))
@click.option('--density', 'opt_density',
  type=cfg.KeyframeMetadataVar,
  default=types.KeyframeMetadata.BASIC,
  help=click_utils.show_help(types.KeyframeMetadata))
@processor
@click.pass_context
def cli(ctx, sink, opt_disk, opt_density):
  """Generates KeyframeStatus metadata"""
  # Recommended: Use Expanded density to check for all keyframes

  # -------------------------------------------------
  # imports

  import os
  from os.path import join
  from pathlib import Path

  from vframe.settings.paths import Paths
  from vframe.settings import vframe_cfg as cfg
  from vframe.utils import file_utils, logger_utils

  from vframe.models.metadata_item import KeyframeStatusMetadataItem
  
  # -------------------------------------------------
  # process

  log = logger_utils.Logger.getLogger()

  # set paths
  media_type = types.Metadata.KEYFRAME
  metadata_type = types.Metadata.KEYFRAME_STATUS
  dir_keyframes = Paths.media_dir(media_type, data_store=opt_disk, 
    verified=ctx.opts['verified'])

  # iterate sink
  while True:
    chair_item = yield
    sha256 = chair_item.sha256
    sha256_tree = file_utils.sha256_tree(sha256)
    dir_parent = join(dir_keyframes, sha256_tree, sha256)
    
    # check if keyframe metadata exists
    keyframe_metadata_item = chair_item.item.get_metadata(types.Metadata.KEYFRAME)
    if not keyframe_metadata_item:
      log.error('no keyframe metadata. try "append -t keyframe", {}'.format(keyframe_metadata_item))
      chair_item.item.set_metadata(metadata_type, {})
    else:
      # check if the keyframes images exist
      status = {k: False for k in cfg.IMAGE_SIZE_LABELS}
      if Path(dir_parent).exists():

        # get keyframe numbers
        idxs = keyframe_metadata_item.get_keyframes(opt_density)

        for idx in idxs:
          labels = [v for k, v in cfg.IMAGE_SIZE_LABELS.items()]
          for k, label in cfg.IMAGE_SIZE_LABELS.items():
            fpp_im = Path(dir_parent, file_utils.zpad(idx), label, 'index.jpg')
            if fpp_im.exists():
              status[k] = True

        # append metadata to chair_item's mapping item
        chair_item.item.set_metadata(metadata_type, KeyframeStatusMetadataItem(status))
    
    
    # -------------------------------------------------
    # continue processing other items

    sink.send(chair_item)
