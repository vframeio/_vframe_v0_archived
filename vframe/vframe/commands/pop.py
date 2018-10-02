"""
Pop/remove metadata from the processor
"""

import click

from vframe.settings import types
from vframe.utils import click_utils
from vframe.settings import vframe_cfg as cfg

from cli_vframe import processor

# --------------------------------------------------------
# Removes metadata attributes
# --------------------------------------------------------
@click.command('pop', short_help='Append metadata to items')
@click.option('-t', '--type', 'opt_metadata_types',
  type=cfg.MetadataVar,
  show_default=True,
  required=True,
  multiple=True,
  help=click_utils.show_help(types.Metadata))
@processor
@click.pass_context
def cli(ctx, sink, opt_metadata_types):
  """Removes metadata attributes"""
  
  # -------------------------------------------------
  # import

  from tqdm import tqdm
  
  from vframe.utils import logger_utils

  
  # -------------------------------------------------
  # process
  
  log = logger_utils.Logger.getLogger()

  # accumulate items
  chair_items = []
  while True:
    try:
      chair_items.append( (yield) )
    except GeneratorExit as ex:
      break

  for opt_metadata_type in opt_metadata_types:
    log.info('removing metadata: {}'.format(opt_metadata_type))
    for chair_item in chair_items:
      chair_item.media_record.remove_metadata(opt_metadata_type)

  
  # -------------------------------------------------
  # rebuild the generator
  
  for chair_item in tqdm(chair_items):
      sink.send(chair_item)
