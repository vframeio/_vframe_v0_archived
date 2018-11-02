"""
Mine negative images for Null class with Yolo training
"""
import click
import pandas as pd

from vframe.utils import click_utils
from vframe.settings import types
from vframe.settings import vframe_cfg as cfg
from vcat.settings import vcat_cfg

# --------------------------------------------------------
# testing
# --------------------------------------------------------
@click.command()
@click.option('-i', '--input', 'opt_fp_neg', required=True,
  help='Negatives CSV')
@click.option('-o', '--output', 'opt_dir_project', required=True,
  help='Path to existing YOLO project')
@click.option('-d', '--disk', 'opt_disk',
  default=click_utils.get_default(types.DataStore.HDD),
  type=cfg.DataStoreVar,
  show_default=True,
  help=click_utils.show_help(types.DataStore))
@click.option('--size', 'opt_size',
  type=cfg.ImageSizeVar,
  default=click_utils.get_default(types.ImageSize.LARGE),
  help=click_utils.show_help(types.ImageSize))
@click.pass_context
def cli(ctx, opt_fp_neg, opt_dir_project, opt_disk, opt_size):
  """Generates negative images"""

  # ------------------------------------------------
  # imports
  import os
  from os.path import join
  from glob import glob
  from pathlib import Path

  from vframe.utils import logger_utils, im_utils, file_utils
  from vframe.settings.paths import Paths

  log = logger_utils.Logger.getLogger()
  log.debug('negative mining')

  dir_media_unver = Paths.media_dir(types.Metadata.KEYFRAME, data_store=opt_disk, verified=types.Verified.UNVERIFIED)
  dir_media_ver = Paths.media_dir(types.Metadata.KEYFRAME, data_store=opt_disk, verified=types.Verified.VERIFIED)
  opt_size_label = cfg.IMAGE_SIZE_LABELS[opt_size]

  fp_train_neg = join(opt_dir_project, vcat_cfg.FP_TRAIN_NEGATIVES)
  dir_labels_negative = join(opt_dir_project, vcat_cfg.DIR_LABELS_NEGATIVE)
  dir_negative = join(opt_dir_project, vcat_cfg .DIR_IMAGES_NEGATIVE)

  file_utils.mkdirs(dir_negative)
  file_utils.mkdirs(dir_labels_negative)
  
  negative_list = pd.read_csv(opt_fp_neg)
  negative_list['description'] = negative_list['description'].fillna('')  # ensure not empty
  # negative_list['desc'] = negative_list['desc'].astype('str') 
  neg_training_files = []

  # for sha256 in sha256_list[:35]:
  for i, row in negative_list.iterrows():
    sha256 = row['sha256']
    sha256_tree = file_utils.sha256_tree(sha256)
    ver_list = glob(join(dir_media_ver, sha256_tree, sha256, "*"))
    unver_list = glob(join(dir_media_unver, sha256_tree, sha256, "*"))
    dir_frames = ver_list + unver_list

    log.debug('adding {} frames about "{}"'.format(len(dir_frames), row['description']))

    for dir_frame in dir_frames:
      frame_idx = Path(dir_frame).stem
      fp_keyframe_src = join(dir_frame, opt_size_label, 'index.jpg')
      fpp_keyframe_src = Path(fp_keyframe_src)
      if fpp_keyframe_src.exists():
        # create symlinked image
        fpp_keyframe_dst = Path(join(dir_negative, '{}_{}.jpg'.format(sha256, frame_idx)))
        if fpp_keyframe_dst.exists() and fpp_keyframe_dst.is_symlink():
          fpp_keyframe_dst.unlink()
        fpp_keyframe_dst.symlink_to(fpp_keyframe_src)
        # create empty label
        fp_label_txt = join(dir_labels_negative, '{}_{}.txt'.format(sha256, frame_idx))
        with open(fp_label_txt, 'w') as fp:
          fp.write('')
        # and, add this file to the training list
        neg_training_files.append(str(fpp_keyframe_dst))


  # for each keyframe if it exists
  log.info('writing {} lines to: {}'.format(len(neg_training_files), fp_train_neg))
  file_utils.write_text(neg_training_files, fp_train_neg)
  
  # add prompt
  log.info('mv labels_negative/*.txt labels/')
  log.info('mv images_negative/*.jpg images/')