"""
Appends metadata to media records
"""

import click

from vframe.utils import click_utils
from vframe.settings import types
from vframe.settings import vframe_cfg as cfg

# --------------------------------------------------------
# Shard a pickle into smaller pices
# --------------------------------------------------------
@click.command()
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
  required=True,
  help=click_utils.show_help(types.Metadata))
@click.option('--verified', 'opt_verified',
  type=cfg.VerifiedVar,
  default=click_utils.get_default(types.Verified.VERIFIED),
  show_default=True,
  help=click_utils.show_help(types.Verified))
@click.option('-n', '--num-pieces', 'opt_num_pieces')
@click.pass_context
def cli(ctx, fp_in, opt_format, opt_disk, opt_metadata_types, opt_verified, opt_num_pieces):
  """Add mappings data to chain"""

  # -------------------------------------------------
  # imports 

  import os
  
  from tqdm import tqdm

  from vframe.settings.paths import Paths  
  from vframe.utils import file_utils, logger_utils
  
  
  # -------------------------------------------------
  # process 

  log = logger_utils.Logger.getLogger()
  log.info('opt_format: {}'.format(opt_format))
  log.info('opt_disk: {}'.format(opt_disk))
  log.info('opt_metadata_type(s): {}'.format(opt_metadata_types))
  
  if not fp_in:
    fps_in = Paths.metadata_index(opt_metadata_type, data_store=opt_disk, 
      file_format=opt_format, verified=opt_verified)

  log.info('fp_in: {}'.format(fp_in))

  # load the file raw
  data = file_utils.lazyload(fp_in)

  