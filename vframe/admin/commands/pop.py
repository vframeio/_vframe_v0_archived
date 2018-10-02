"""
Appends metadata to media records
"""

import click

from vframe.utils import click_utils
from vframe.settings import types
from vframe.settings import vframe_cfg as cfg

# --------------------------------------------------------
# Add source items to chain
# Sources can be original mappings or metadata, which always include mappings
# --------------------------------------------------------
@click.command('append', short_help='Append metadata to items')
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
@click.option('-t', '--type', 'opt_metadata_types',
  default=None, # default uses original mappings
  type=cfg.MetadataVar,
  show_default=True,
  multiple=True,
  required=True,
  help=click_utils.show_help(types.Metadata))
@click.pass_context
def cli(ctx, fp_in, opt_format, opt_disk, opt_metadata_types):
  """Add mappings data to chain"""

  # -------------------------------------------------
  # imports 

  import os
  
  from tqdm import tqdm

  from vframe.settings.paths import Paths  
  from vframe.utils import file_utils, logger_utils
  from vframe.models.chair_item import ChairItem
  from vframe.models.metadata_item import KeyframeMetadataItem, MediainfoMetadataItem
  from vframe.models.metadata_item import KeyframeStatusMetadataItem, ClassifyMetadataItem
  from vframe.models.metadata_item import FeatureMetadataItem, DetectMetadataItem
  
  
  # -------------------------------------------------
  # process 

  log = logger_utils.Logger.getLogger()
  log.info('opt_format: {}'.format(opt_format))
  log.info('opt_disk: {}'.format(opt_disk))
  log.info('opt_metadata_type(s): {}'.format(opt_metadata_types))
  
  if not fp_in:
    fps_in = [Paths.metadata_index(opt_metadata_type, data_store=opt_disk, file_format=opt_format,
      verified=ctx.opts['verified']) for opt_metadata_type in opt_metadata_types]

  # accumulate items
  chair_items = []
  while True:
    try:
      chair_items.append( (yield) )
    except GeneratorExit as ex:
      break

  # append items
  for opt_metadata_type, fp_in in zip(opt_metadata_types, fps_in):

    log.debug('opening: {}'.format(fp_in))
    media_records = file_utils.load_records(fp_in)
    
    if not media_records:
      log.error('no metadata items, or file missing')
      log.error('check the "-d" / "--disk" location and try again')
      return

    log.debug('appending: {}'.format(opt_metadata_type.name.lower()))

    for chair_item in tqdm(chair_items):
      sha256 = chair_item.sha256
      metadata = media_records[sha256].get_metadata(opt_metadata_type)
      chair_item.media_record.set_metadata(opt_metadata_type, metadata)


  # ------------------------------------------------
  # rebuild the generator
  for chair_item in tqdm(chair_items):
      sink.send(chair_item)
