"""
Use Darknet for classification
"""
import click

from vframe.settings import types
from vframe.utils import click_utils
from vframe.settings import vframe_cfg as cfg

from cli_vframe import processor


@click.command()
@click.option('-t', '--net-type', 'opt_net',
  default=click_utils.get_default(types.ClassifyNet.PLACES365),
  type=cfg.ClassifyNetVar,
  help=click_utils.show_help(types.ClassifyNet))
@click.option('-d', '--disk', 'opt_disk',
  default=click_utils.get_default(types.DataStore.SSD),
  type=cfg.DataStoreVar,
  show_default=True,
  help=click_utils.show_help(types.DataStore))
@processor
@click.pass_context
def cli(ctx, sink, opt_net, opt_disk):
  """Generates classification metadata (Darknet)"""

  # -------------------------------------------------------------
  # imports
  
  import os
  from os.path import join
  from pathlib import Path

  import cv2 as cv
  import numpy as np

  from vframe.utils import file_utils, im_utils, logger_utils
  from vframe.models.metadata_item import ClassifyMetadataItem, ClassifyResult

  # -------------------------------------------------------------
  # init

  log = logger_utils.Logger.getLogger()
  log.error('not yet implemented')
  return
  
  while True:
  
    chair_item = yield

    # send back to generator
    sink.send(chair_item)