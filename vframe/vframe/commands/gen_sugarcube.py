"""
Generates Sugarcube metadata using files provided by client
"""

import click

from vframe.settings import types
from vframe.utils import click_utils
from vframe.settings import vframe_cfg as cfg

from cli_vframe import processor

@click.command('gen_sugarcube', short_help='Generate Sugarcube metadata')
@click.option('-i', '--input', 'fp_in', required=True, type=str,
    help="Path to mappings CSV files from sugarcube/littlefork")
@click.option('-d', '--disk', 'opt_disk',
  default=click_utils.get_default(types.DataStore.SSD),
  type=cfg.DataStoreVar,
  show_default=True,
  help=click_utils.show_help(types.DataStore))
@click.option('--media-format', 'opt_media_type',
  default=types.MediaFormat.VIDEO.name.lower(),
  type=cfg.MediaFormatVar,
  show_default=True,
  help=click_utils.show_help(types.MediaFormat))
@click.option('--minify/--no-minify', 'opt_minify', default=False,
  help='Minify output if using JSON')
@click.option('--all/--filter-status', 'opt_all', is_flag=True, default=False,
  help='Keep all or filter by verified status')
@click.option('-f', '--force', 'opt_force', is_flag=True,
  help='Force overwrite')
@processor
@click.pass_context
def cli(ctx, sink, fp_in, opt_disk, opt_media_type, opt_minify, opt_all, opt_force):
  """Generates Sugarcube metadata"""
  
  # -------------------------------------------------
  # imports

  import os
  from os.path import join
  from pathlib import Path

  from tqdm import tqdm
  
  from vframe.utils import file_utils, logger_utils
  from vframe.settings.paths import Paths
  from vframe.settings import vframe_cfg as cfg
  from vframe.models.metadata_item import SugarcubeMetadataItem
  from vframe.models.chair_item import ChairItem


  # -------------------------------------------------
  # process here
  
  log = logger_utils.Logger.getLogger()
  log.debug('fp_in: {}'.format(fp_in))

  # accumulate chair items
  chair_items = []
  while True:
    try:
      chair_items.append( (yield) )
    except GeneratorExit as ex:
      break

  csv_rows = file_utils.load_csv(fp_in) # as list

  # load sugarcube data into dict`
  sugarcube_data = {}
  log.debug('loading sugarcube data')
  for row in tqdm(csv_rows):
    sha256 = row.get('sha256', None)
    if sha256:
      sugarcube_data[sha256] = row
    
  # iterate media_records to append SugarcubeMetadataItems
  for chair_item in chair_items:
    media_record = chair_item.media_record
    sha256 = media_record.sha256
    # sa_id,sha256,md5,location,verified
    fp_media = sugarcube_data[sha256].get('location', None)
    sa_id = sugarcube_data[sha256].get('sa_id', None)
    md5 = sugarcube_data[sha256].get('md5', None)
    metadata = SugarcubeMetadataItem(fp_media, sa_id, md5)
    chair_item.media_record.set_metadata(types.Metadata.SUGARCUBE, metadata)
  
  
  # -------------------------------------------------
  # rebuild generator

  for chair_item in chair_items:
    sink.send(chair_item)
