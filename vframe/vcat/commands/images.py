"""This opens images to create generator
"""
from os.path import join
import json
import urllib
from urllib.request import urlretrieve
from pathlib import Path

import click
from tqdm import tqdm
import workerpool  # pip install git+https://github.com/shazow/workerpool

from vframe.utils import logger_utils, file_utils
from vcat.utils import vcat_utils
from vcat.settings import vcat_cfg


# --------------------------------------------------------
# Download images from VCAT
# --------------------------------------------------------
@click.command()
@click.option('-i', '--input', 'opt_fp_in',
  default=vcat_cfg.FP_VCAT_ANNOTATIONS,
  help='VCAT API JSON file')
@click.option('-e', '--exclude', 'opt_excludes', 
  type=int, multiple=True,
  help='Classes to exclude')
@click.option('-o', '--output', 'opt_dir_out', 
  default=vcat_cfg.DIR_IMAGES,
  help='Image output directory')
@click.option('--s3', 'opt_s3_url', envvar='VCAT_S3_URL', required=True,
  help='S3 VCAT URL')
@click.option('--threads', 'opt_nthreads', default=10,
  type=click.IntRange(0,100),
  help='Number of threads')
@click.pass_context
def cli(ctx, opt_fp_in, opt_excludes, opt_dir_out, opt_s3_url, opt_nthreads):
  """Download only the class hierarcy from VCAT API"""

  log = logger_utils.Logger.getLogger()
  log.debug('download images')
  if not opt_s3_url:
    log.error('S3 URL required. Try source env variables')
    return

  # get the ordered hierarchy
  vcat_data = vcat_utils.load_annotations(opt_fp_in, opt_excludes)
  hierarchy_tree = vcat_utils.hierarchy_tree(vcat_data['hierarchy'].copy())

  # build image ID lookup table. the regions refer to these
  image_lookup =  {}
  for vcat_class_id, object_class in vcat_data['object_classes'].items():
    for image in object_class['images']:
      image_lookup[image['id']] = image
  
  url_maps = []
  for vcat_class_id, object_class in vcat_data['object_classes'].items():
    for region in object_class['regions']:
      im_meta = image_lookup[region['image']]
      url = vcat_utils.format_im_url(opt_s3_url, im_meta)
      # log.info(url)
      fp_out = vcat_utils.format_im_fn(im_meta)
      fp_out = join(opt_dir_out, fp_out)
      url_maps.append( {'url': url, 'fp_out': fp_out})
  
  if not Path(opt_dir_out).exists():
    file_utils.mkdirs(opt_dir_out)

  
  # download pool
  global pbar
  pbar = tqdm(total=len(url_maps))
  pool = workerpool.WorkerPool(size=opt_nthreads)
  pool.map(downloader, url_maps)
  # Send shutdown jobs to all threads
  #   and wait until all the jobs have been completed
  pool.shutdown()
  pool.wait()
  pbar.close()



# --------------------------------------------------------
# aux functions
# --------------------------------------------------------
pbar = None  # global scope

def downloader(url_map):
  global pbar, log
  pbar.update(1)
  try:
    if not Path( url_map['fp_out']).exists():
      urlretrieve(url_map['url'], url_map['fp_out'])
  except:
    log.debug('could not download: {}'.format(url_map['url']))
