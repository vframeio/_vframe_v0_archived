"""This opens images to create generator
"""
import csv
from collections import OrderedDict
from operator import itemgetter
from pprint import pprint
import itertools

import click

from vframe.utils import click_utils
from vframe.settings import types
from vframe.utils import logger_utils, im_utils, file_utils
from vcat.utils import vcat_utils
from vcat.models.yolo_item import YoloAnnoItem

FP_VCAT = '/data_store_ssd/datasets/syrianarchive/v1_annotations/20180819.json'
DIR_PROJECT = '/data_store_ssd/apps/vframe/models/darknet/vframe/test_00/'
FP_OUT_HIERARCHY = '/data_store_ssd/datasets/syrianarchive/v1_annotations/hierarchy.txt'
FP_OUT_FLAT = '/data_store_ssd/datasets/syrianarchive/v1_annotations/flat.txt'

# --------------------------------------------------------
# testing
# --------------------------------------------------------
@click.command()
@click.option('-i', '--input', 'fp_in', 
  default=FP_VCAT,
  help='Override file input path')
@click.option('--images', 'dir_images',
  help='Path to project directory')
@click.option('-e', '--exclude', 'opt_excludes', 
  type=int, multiple=True,
  help='Classes to exclude')
@click.option('-p', '--parent', 'opt_parent_hierarchy', 
  is_flag=True, default=False,
  help='Use hierarchical parent labeling')
@click.pass_context
def cli(ctx, fp_in, dir_images, opt_excludes, opt_parent_hierarchy):
  """Generate clases and labels file"""

  # ------------------------------------------------
  # imports

  log = logger_utils.Logger.getLogger()
  log.debug('generate classes')

  log.debug('loading: {}'.format(fp_in))
  vcat_data = vcat_utils.load_annotations(fp_in, opt_excludes)
  
  
  

  yolo_annos = yolo_utils.create_annos()
