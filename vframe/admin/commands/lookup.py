"""
Looks up info about Sugarcube item
"""
import click

from vframe.settings import types
from vframe.utils import click_utils
from vframe.settings import vframe_cfg as cfg

from cli_admin import processor

@click.command('search')
@click.option('--type', 'opt_type', required=True,
  type=cfg.SearchParamVar,
  default=click_utils.get_default(types.SearchParam.SHA256),
  help=click_utils.show_help(types.SearchParam))
@click.argument('arg_id')
@processor
@click.pass_context
def cli(ctx, sink, opt_type, arg_id):
  """search for info with ID"""

  # -------------------------------------------------
  # imports

  import os
  from os.path import join
  from pathlib import Path

  from vframe.utils import file_utils, logger_utils
  from vframe.settings.paths import Paths
  from vframe.settings import vframe_cfg as cfg

  # -------------------------------------------------
  # process 

  log = logger_utils.Logger.getLogger()
  log.debug('opt_type: {}, arg_id: {}'.format(opt_type, arg_id))
  
  chair_items = []
  while True:
    try:
      chair_items.append( (yield) )
    except GeneratorExit as ex:
      break

  # iterate through all chair items loooking for the search param
  chair_items_copy = chair_items.copy()
  chair_items = []

  log.info('searching {:,} media records for {}: {}'.format(len(chair_items_copy), opt_type, arg_id))

  for chair_item in chair_items_copy:
    media_record = chair_item.media_record
    sha256 = media_record.sha256

    if opt_type == types.SearchParam.SHA256:
      # quick match sha256
      if arg_id == sha256:
        chair_items.append(chair_item)
        break
    else:
      # get sc metadata
      sugarcube_metadata = media_record.get_metadata(types.Metadata.SUGARCUBE)
      
      if not sugarcube_metadata:
        log.error('no sugarcube metadata. Try "append -t sugarcube"')
        return

      # match other params
      if opt_type == types.SearchParam.SA_ID:
        if arg_id == sugarcube_metadata.sa_id:
          chair_items.append(chair_item)
          break 
      elif opt_type == types.SearchParam.MD5:
        if arg_id == sugarcube_metadata.md5:
          chair_items.append(chair_item)
          break 

  if not len(chair_items) > 0:
    log.error('No results')
  else:
    log.info('{} item found'.format(len(chair_items)))

  # -------------------------------------------------
  # rebuild generator 

  for chair_item in chair_items:
    sink.send(chair_item)