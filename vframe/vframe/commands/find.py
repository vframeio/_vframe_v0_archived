"""
Finds ID and passes only that media_record to pipeline
future version may enable more grep-like features
"""

import click

from vframe.utils import click_utils
from vframe.settings import types
from vframe.settings import vframe_cfg as cfg

from cli_vframe import processor


# --------------------------------------------------------
# Add source items to chain
# --------------------------------------------------------
@click.command()
@click.option('--type', 'opt_search_type', required=True,
  type=cfg.SearchParamVar,
  default=click_utils.get_default(types.SearchParam.SHA256),
  help=click_utils.show_help(types.SearchParam))
@click.option('--id', 'opt_id', required=True,
  help='ID value to search for')
@processor
@click.pass_context
def cli(ctx, sink, opt_search_type, opt_id):
  """Isolates media record ID"""


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

  log.info('find type: {}, value: {}'.format(opt_search_type, opt_id))

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

  if opt_search_type == types.SearchParam.SHA256:
    
    for chair_item in chair_items_copy:
      media_record = chair_item.media_record
      if media_record.sha256 == opt_id:
        chair_items.append(chair_item)
  else:
    log.error('{} not yet implemented'.format(opt_search_type))
    return

  log.debug('found {} item'.format(len(chair_items)))

  # -------------------------------------------------
  # rebuild the generator

  for item in tqdm(chair_items):
      sink.send(item)