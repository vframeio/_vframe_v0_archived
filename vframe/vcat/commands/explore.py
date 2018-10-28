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
from vcat.settings import vcat_cfg
from vframe.utils import logger_utils, im_utils, file_utils
from vcat.utils import vcat_utils, yolo_utils
from vcat.models.yolo_item import YoloAnnoItem

# --------------------------------------------------------
# testing
# --------------------------------------------------------
@click.command()
@click.option('-i', '--input', 'fp_in', 
  default=vcat_cfg.FP_VCAT_ANNOTATIONS,
  help='VCAT API JSON file')
@click.option('-e', '--exclude', 'opt_excludes', 
  type=int, multiple=True,
  help='Classes to exclude')
@click.option('-p', '--parent', 'opt_parent_hierarchy', 
  is_flag=True, default=False,
  help='Use hierarchical parent labeling')
@click.pass_context
def cli(ctx, fp_in, opt_excludes, opt_parent_hierarchy):
  """Generate clases and labels file"""

  # ------------------------------------------------
  # imports

  log = logger_utils.Logger.getLogger()
  log.debug('generate classes')

  vcat_data = vcat_utils.load_annotations(fp_in, opt_excludes)

  hierarchy_tree = vcat_utils.hierarchy_tree(vcat_data['hierarchy'])
  h_tree_display = vcat_utils.hierarchy_tree_display(hierarchy_tree)
  h_tree_flat = vcat_utils.hierarchy_flat(hierarchy_tree)
  from pprint import pprint
  log.info(70*'-')
  log.debug(h_tree_display)
  log.info(70*'-')
  for k, v in h_tree_flat.items():
    log.debug('{label_index}: "{display_name}" has {region_count} annos'.format(**v))