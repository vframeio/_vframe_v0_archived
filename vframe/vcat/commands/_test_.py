"""This opens images to create generator
"""
import click

from vframe.utils import click_utils
from vframe.settings import types

# --------------------------------------------------------
# testing
# --------------------------------------------------------
@click.command('_test_', short_help='Test command')
@click.option('-i', '--input', 'fp_in', default=None,
  help='Override file input path')
@click.pass_context
def cli(ctx, fp_in):
  """Add mappings data to chain"""

  # ------------------------------------------------
  # imports
  from vframe.utils import logger_utils, im_utils, file_utils
  
  log = logger_utils.Logger.getLogger()
  log.debug('test')
  