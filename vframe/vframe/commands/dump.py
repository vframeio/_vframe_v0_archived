"""
Dumps data every N intervals
"""
import os
from os.path import join
from pathlib import Path
# import collections

import click

from vframe.settings import vframe_cfg as cfg
from vframe.settings import types
from vframe.settings.paths import Paths
from vframe.utils import file_utils, click_utils
from vframe.utils.logger_utils import Logger

from cli_vframe import processor

# --------------------------------------------------------
# Save metadata to JSON/Pickle
# --------------------------------------------------------
@click.command()
@click.option('-o', '--output', 'fp_out', default=None,
  help='Path to output file (overrides other settings)')
@click.option('-e', '--ext', 'opt_format',
  default=click_utils.get_default(types.FileExt.PKL),
  type=cfg.FileExtVar,
  show_default=True,
  help=click_utils.show_help(types.FileExt))
@click.option('-d', '--disk', 'opt_disk',
  default=click_utils.get_default(types.DataStore.SSD),
  type=cfg.DataStoreVar,
  show_default=True,
  help=click_utils.show_help(types.DataStore))
@click.option('-t', '--type', 'opt_metadata_type',
  type=cfg.MetadataVar,
  required=True,
  multiple=True,
  show_default=True,
  help=click_utils.show_help(types.Metadata))
@click.option('--minify/--no-minify', 'opt_minify', default=False,
  help='Minify output if using JSON')
@click.option('-f', '--force', 'opt_force', is_flag=True,
  help='Force overwrite')
@click.option('--interval', 'opt_interval', required=True, type=int,
  help='Save/dump interval')
@processor
@click.pass_context
def cli(ctx, sink, fp_out, opt_format, opt_disk, opt_metadata_type, opt_minify, 
  opt_force, opt_interval):
  """Writes items to disk as JSON or Pickle"""

  from vframe.utils import logger_utils
  
  log = logger_utils.Logger.getLogger()

  # construct path
  if not fp_out:
    fp_out = Paths.metadata_index(opt_metadata_type, data_store=opt_disk, 
      file_format=opt_format, verified=ctx.opts['verified'])

  # accumulate items
  chair_items = []
  interval_count = 0

  while True:

    chair_item = yield
    chair_items.append(chair_item)

    if len(chair_items) >= opt_interval:
      # save data
      log.debug('chair_items: {}'.format(len(chair_items)))
      mapping_items = file_utils.chair_to_mapping(chair_items)
      log.debug('mapping_items: {}'.format(len(mapping_items)))
      
      # checkpoint filename
      fpp_out = Path(fp_out)
      ckpt_suffix = '{}_{}'.format(interval_count*opt_interval, ((interval_count+1)*opt_interval))
      fp_out = join(str(fpp_out.parent), '{}_{}{}'.format(fpp_out.stem, ckpt_suffix, fpp_out.suffix))

      # # write to disk
      log.debug('saving checkpoint to: {}'.format(fp_out))
      file_utils.write_serialized_items(mapping_items, fp_out, ensure_path=True, minify=opt_minify)
      
      # dump data
      interval_count += 1

      for chair_item in chair_items:
        chair_item.media_record.remove_metadata(opt_metadata_type)

      chair_items = []

    sink.send(chair_item)