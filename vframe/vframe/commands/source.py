"""
Source is a generator and is used at the beginning of the processor workflows
"""

import click

from vframe.utils import click_utils
from vframe.settings import types
from vframe.settings import vframe_cfg as cfg

from cli_vframe import generator

# --------------------------------------------------------
# Add source items to chain
# Sources can be original mappings or metadata, which always include mappings
# --------------------------------------------------------
@click.command('source', short_help='Add media records items to chain')
@click.option('-i', '--input', 'fp_in', default=None,
  help='Override file input path')
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
@generator
@click.pass_context
def cli(ctx, sink, fp_in, opt_format, opt_disk):
  """Add mappings data to chain"""
  
  import os
  import logging

  from tqdm import tqdm

  from vframe.settings.paths import Paths
  from vframe.utils import file_utils, logger_utils
  from vframe.models.chair_item import MediaRecordChairItem
  from vframe.utils import logger_utils
  
  
  log = logger_utils.Logger.getLogger()

  log.info('opt_format: {}'.format(opt_format))
  log.info('opt_disk: {}'.format(opt_disk))

  if not fp_in:
    fp_in = Paths.media_record_index(data_store=opt_disk, file_format=opt_format, 
      verified=ctx.opts['verified'])

  # load mappings
  # TODO make multithreaded
  log.info('opening: {}'.format(fp_in))
  media_records = file_utils.load_records(fp_in)

  # update ctx variable
  ctx.opts['num_items'] = len(media_records)
  # ctx.opts['chair_type'] = ChairItemType.MEDIA_RECORD
  
  # begin processing
  if not media_records or not ctx.opts['num_items'] > 0:
    log.error('no media_record available to process')
    return
  else:
    log.info('dispatching {:,} records...'.format(ctx.opts['num_items']))
    for sha256, media_record in tqdm(media_records.items()):
      sink.send( MediaRecordChairItem(ctx, media_record) )
