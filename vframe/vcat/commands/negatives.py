"""
Mine negative images for Null class with Yolo training
"""
import click

from vframe.utils import click_utils
from vframe.settings import types

FP_NEG_LIST = '/data_store_ssd/apps/vframe/models/darknet/vframe/cluster_munition_07/negatives.txt'
DIR_PROJECT = '/data_store_ssd/apps/vframe/models/darknet/vframe/cluster_munition_07'

# --------------------------------------------------------
# testing
# --------------------------------------------------------
@click.command()
@click.option('-i', '--input', 'fp_neg_list', default=FP_NEG_LIST,
  help='Override file input path')
@click.option('--project', 'dir_project', default=DIR_PROJECT,
  help='Override file input path')
@click.option('-v', '--verified', 'opt_verified',
  default=click_utils.get_default(types.Verified.VERIFIED),
  type=click_utils.VerifiedVar,
  show_default=True,
  help=click_utils.show_help(types.Verified))
@click.option('-d', '--disk', 'opt_disk',
  default=click_utils.get_default(types.DataStore.SSD),
  type=click_utils.DataStoreVar,
  show_default=True,
  help=click_utils.show_help(types.DataStore))
@click.option('--density', 'opt_density',
  default=click_utils.get_default(types.KeyframeMetadata.EXPANDED),
  show_default=True,
  type=click_utils.KeyframeVar,
  help=click_utils.show_help(types.KeyframeMetadata))
@click.option('--size', 'opt_size',
  type=click_utils.ImageSizeVar,
  default=click_utils.get_default(types.ImageSize.LARGE),
  help=click_utils.show_help(types.ImageSize))
@click.pass_context
def cli(ctx, fp_neg_list, dir_project, opt_verified, opt_disk, opt_density, opt_size):
  """Generates negative images"""

  # ------------------------------------------------
  # imports
  import os
  from os.path import join
  from glob import glob
  from pathlib import Path

  from vframe.settings import vframe_cfg as cfg
  from vframe.utils import logger_utils, im_utils, file_utils
  from vframe.settings.paths import Paths

  log = logger_utils.Logger.getLogger()
  log.debug('negative mining')

  dir_media = Paths.media_dir(types.Metadata.KEYFRAME, data_store=opt_disk, verified=opt_verified)
  opt_size_label = cfg.IMAGE_SIZE_LABELS[opt_size]

  sha256_list = file_utils.load_text(fp_neg_list)
  
  fp_train_neg = join(dir_project, 'train_negative.txt')
  dir_labels_negative = join(dir_project, 'labels_negative')
  dir_negative = join(dir_project, 'images_negative')

  file_utils.mkdirs(dir_negative)
  file_utils.mkdirs(dir_labels_negative)
  
  neg_training_files = []

  for sha256 in sha256_list[:35]:
    log.debug('sha256: {}'.format(sha256))
    sha256_tree = file_utils.sha256_tree(sha256)
    dir_sha256 = join(dir_media, sha256_tree, sha256)
    frame_idxs = os.listdir(dir_sha256)

    for frame_idx in frame_idxs:
      log.debug('frame: {}'.format(frame_idx))
      fp_keyframe_src = join(dir_sha256, file_utils.zpad(frame_idx), opt_size_label, 'index.jpg')
      fpp_keyframe_src = Path(fp_keyframe_src)
      if fpp_keyframe_src.exists():
        log.debug('exists: {}'.format(fpp_keyframe_src))
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
  file_utils.write_text(neg_training_files, fp_train_neg)