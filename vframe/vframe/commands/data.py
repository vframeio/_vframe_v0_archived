"""
Add/remove metadata to media records
"""

import click

from vframe.utils import click_utils
from vframe.settings import types
from vframe.settings import vframe_cfg as cfg

from cli_vframe import processor


# --------------------------------------------------------
# Add/remove metadata objects to chair items
# --------------------------------------------------------
@click.command()
@click.option('-a', '--action', 'opt_action', required=True,
  type=cfg.ActionVar,
  default=click_utils.get_default(types.Action.ADD),
  help=click_utils.show_help(types.Action))
@click.option('-i', '--input', 'fp_in', default=None,  #  metadata can be filepath
  help='Override file input path')
@click.option('-e', '--ext', 'opt_format',  # shortcut for json, pkl (no csv)
  default=click_utils.get_default(types.FileExt.PKL),
  type=cfg.FileExtVar,
  show_default=True,
  help=click_utils.show_help(types.FileExt))
@click.option('-d', '--disk', 'opt_disk',  # which storage device to use
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
@processor
@click.pass_context
def cli(ctx, sink, opt_action, fp_in, opt_format, opt_disk, opt_metadata_types):
  """Add/remove metadata to chair"""
  
  # -------------------------------------------------
  # imports 

  import os
  
  from tqdm import tqdm

  from vframe.settings.paths import Paths
  from vframe.utils import file_utils, logger_utils
  from vframe.models.chair_item import ChairItem
  
  
  # -------------------------------------------------
  # process 

  log = logger_utils.Logger.getLogger()
  log.info('opt_action: {}'.format(opt_action))
  log.info('fp_in: {}'.format(fp_in))
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


  if opt_action == types.Action.ADD:
    
    # ------------------------------------------------------------------------
    # append items

    for opt_metadata_type, fp_in in zip(opt_metadata_types, fps_in):

      log.debug('opening: {}'.format(fp_in))
      media_records = file_utils.load_records(fp_in)
      
      if not media_records:
        log.error('no metadata items or file. check "-d" / "--disk" and try again')
        return

      log.debug('appending: {}'.format(opt_metadata_type.name.lower()))

      for chair_item in tqdm(chair_items):
        sha256 = chair_item.sha256
        metadata = media_records[sha256].get_metadata(opt_metadata_type)
        chair_item.media_record.set_metadata(opt_metadata_type, metadata)

  elif opt_action == types.Action.RM:
    
    # ------------------------------------------------------------------------
    # remove metadata items

    for opt_metadata_type in opt_metadata_types:
      
      log.info('removing metadata: {}'.format(opt_metadata_type))

      for chair_item in chair_items:
        chair_item.media_record.remove_metadata(opt_metadata_type)


  # ------------------------------------------------
  # rebuild the generator
  for chair_item in tqdm(chair_items):
      sink.send(chair_item)
