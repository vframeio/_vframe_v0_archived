"""
Prints metadata about items in processor
- useful to combine with "slice"
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
@click.command('display', short_help='Display info')
@click.option('-m', '--metadata', 'opt_metadata', is_flag=True, default=True,
  help='Show all metadata objects')
@processor
@click.pass_context
def cli(ctx, sink, opt_metadata):
  """Displays info for debugging"""

  from vframe.utils import logger_utils
  
  log = logger_utils.Logger.getLogger()

  # accumulate items
  chair_items = []
  while True:
    try:
      chair_items.append( (yield) )
    except GeneratorExit as ex:
      break

  # -----------------------------------------------------------------
  # do something here
  from pprint import pprint
  # rebuild the generator
  for chair_item in chair_items:
    media_record = chair_item.media_record
    metadata_records = media_record.metadata
    log.debug('sha256: {}'.format(media_record.sha256))
    log.debug('\tformat: {}'.format(media_record.media_format))
    log.debug('\tverified: {}'.format(media_record.verified))
    if opt_metadata:
      for metadata_type, metadata_obj in metadata_records.items():
        log.debug('\ttype: {}'.format(metadata_type))
        try:
          log.debug('\tmetadata: {}'.format(metadata_obj.serialize()))
        except Exception as ex:
          log.debug('\tmetadata: {}'.format(metadata_obj.__dict__))

  # -----------------------------------------------------------------

  
  # rebuild the generator
  for chair_item in chair_items:
    sink.send(chair_item)