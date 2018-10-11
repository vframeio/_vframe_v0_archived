"""
Add/remove metadata to media records
"""

import click

from vframe.utils import click_utils
from vframe.settings import types
from vframe.settings import vframe_cfg as cfg

from cli_vframe import processor


# --------------------------------------------------------
# purges media record
# --------------------------------------------------------
@click.command()
@processor
@click.pass_context
def cli(ctx, sink):
  """Purges media record data to free up RAM"""
  
  # -------------------------------------------------
  # imports 
  
  from tqdm import tqdm

  from vframe.utils import file_utils, logger_utils
  
  
  # -------------------------------------------------
  # init 

  log = logger_utils.Logger.getLogger()


  # -------------------------------------------------
  # process 
  
  while True:
    chair_item = yield
    #chair_item.media_record.purge_metadata()
    chair_item.remove_keyframes()
    chair_item.remove_drawframes()
    sink.send(chair_item)