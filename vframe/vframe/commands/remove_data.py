"""
Add/remove metadata to media records
"""

import click

from vframe.utils import click_utils
from vframe.settings import types
from vframe.settings import vframe_cfg as cfg

from cli_vframe import processor


# --------------------------------------------------------
# remove metadata objects from chair items
# --------------------------------------------------------
@click.command()
@click.option('--inline/--no-inline', 'opt_inline', is_flag=True, default=True,
  help='Process items inline or all at once')
@click.option('-t', '--type', 'opt_metadata_types',
  default=None, # default uses original mappings
  type=cfg.MetadataVar,
  show_default=True,
  multiple=True,
  required=True,
  help=click_utils.show_help(types.Metadata))
@processor
@click.pass_context
def cli(ctx, sink, opt_inline, opt_metadata_types):
  """Removes metadata"""
  
  # -------------------------------------------------
  # imports 
  
  from tqdm import tqdm

  from vframe.utils import file_utils, logger_utils
  
  
  # -------------------------------------------------
  # process 

  log = logger_utils.Logger.getLogger()
  log.info('opt_metadata_type(s): {}'.format(opt_metadata_types))
  
  if opt_inline:
    
    while True:
      chair_item = yield
      
      for opt_metadata_type in opt_metadata_types:
        #log.debug('remove {} for {}'.format(opt_metadata_type.name.lower(), chair_item.sha256))
        chair_item.media_record.remove_metadata(opt_metadata_type)

      sink.send(chair_item)

  else:
    # accumulate items
    chair_items = []
    
    while True:
      try:
        chair_items.append( (yield) )
      except GeneratorExit as ex:
        break

    # ------------------------------------------------------------------------
    # remove items

    for opt_metadata_type in opt_metadata_types:
  
      log.info('removing metadata: {}'.format(opt_metadata_type))

      for chair_item in chair_items:
        chair_item.media_record.remove_metadata(opt_metadata_type)

    # ------------------------------------------------
    # rebuild the generator
    for chair_item in tqdm(chair_items):
        sink.send(chair_item)
