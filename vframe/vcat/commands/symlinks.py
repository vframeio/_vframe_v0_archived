"""This opens images to create generator
"""
import click

from vframe.utils import click_utils
from vframe.settings import types
from vframe.utils import logger_utils, im_utils, file_utils

# --------------------------------------------------------
# testing
# --------------------------------------------------------
@click.command('gen_symlinks', short_help='Generate image symlinks')
@click.option('-i', '--input', 'fp_in', default=None,
  help='Override file input path')
@click.pass_context
def cli(ctx, dir_media):
  """Generate image symlinks"""

  # ------------------------------------------------
  # imports
  
  log = logger_utils.Logger.getLogger()
  log.debug('parse train.txt')
  # for each file
  # make filepath
  # create symlink in images directory

  
  """
  for fn_key, image_obj in image_anno_index.items():
    src = image_obj['filepath'] # path to local file
    dst = os.path.join(dir_project_images,'{}{}'.format(fn_key,image_obj['ext']))
    # remove symlink if exists
    if os.path.isfile(dst):
        os.remove(dst)
    # add symlink
    os.symlink(src,dst)
  """