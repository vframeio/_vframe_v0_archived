"""
Example file for processor command
"""

import click

from vframe.settings import types
from vframe.utils import click_utils

from cli_vframe import processor


@click.command('template')
@processor
@click.pass_context
def cli(ctx, sink):
  """[blank template]"""

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
  log.info('test')
  log.debug('test')
  log.warn('test')
  log.error('test')
  log.critical('test')

  chair_items = []
  while True:
    try:
      chair_items.append( (yield) )
    except GeneratorExit as ex:
      break


  # -------------------------------------------------
  # rebuild generator 

  for chair_item in chair_items:
    sink.send(chair_item)