"""This opens images to create generator
"""
import json

import click

from vframe.utils import click_utils
from vframe.settings import types
from vframe.utils import logger_utils, im_utils, file_utils
from vcat.utils import vcat_api


# --------------------------------------------------------
# testing
# --------------------------------------------------------
@click.command()
@click.option('-o', '--output', 'fp_out', required=True,
  help='Path to JSON output file')
@click.option('--username', 'opt_un', envvar='VCAT_USERNAME')
@click.option('--password', 'opt_pw', envvar='VCAT_PASSWORD')
@click.option('-t', '--type', 'opt_get_type', default='all',
  type=click.Choice(['all','hierarchy','class']),
  help="Download options")
@click.option('--id', 'class_id', default=None, type=str,
  help='Class ID to download')
@click.pass_context
def cli(ctx, fp_out, opt_un, opt_pw, opt_get_type, class_id):
  """Download only the class hierarcy from VCAT API"""

  log = logger_utils.Logger.getLogger()
  log.debug('generate classes')
  
  api = vcat_api.API(opt_un, opt_pw)
  
  if opt_get_type == 'hierarchy':
      # downloads only the hierarcy of all VCAT classes
      log.debug('getting hierarcy...')
      vcat_data = api.get_hierarchy()
  elif opt_get_type == 'class':
      # downloads class ID annotations into single JSON object VCAT API
      if not class_id :
        log.error('"--id" is required')
        return
      log.debug('getting class {}...'.format(class_id))
      vcat_data = api.get_class(class_id)
  elif opt_get_type == 'all':
      # download hierarchy and all annotations
      log.debug('getting full annotation file...')
      vcat_data = api.get_full()

  # write JSON to file or stdout
  file_utils.write_json(vcat_data, fp_out, minify=False, 
    ensure_path=True, sort_keys=True)
