"""
Generates CNN feature vectors using PyTorch
"""

import click

from vframe.settings import types
from vframe.utils import click_utils
from vframe.settings import vframe_cfg as cfg

from cli_vframe import processor


@click.command('feature_extractor')
@click.option('-d', '--disk', 'opt_disk',
  default=click_utils.get_default(types.DataStore.SSD),
  type=cfg.DataStoreVar,
  show_default=True,
  help=click_utils.show_help(types.DataStore))
@click.option('--density', 'opt_density',
  default=click_utils.get_default(types.KeyframeMetadata.EXPANDED),
  show_default=True,
  type=cfg.KeyframeMetadataVar,
  help=click_utils.show_help(types.KeyframeMetadata))
@click.option('-t', '--net-type', 'opt_net',
  default=click_utils.get_default(types.PyTorchNet.RESNET18),
  type=cfg.PyTorchNetVar,
  help=click_utils.show_help(types.PyTorchNet))
@click.option('--size', 'opt_size',
  type=cfg.ImageSizeVar,
  default=click_utils.get_default(types.ImageSize.MEDIUM),
  help=click_utils.show_help(types.ImageSize))
@click.option('--gpu', 'opt_gpu', type=int, default=0,
  help='GPU index (use -1 for CPU)')
@processor
@click.pass_context
def cli(ctx, sink, opt_disk, opt_density, opt_net, opt_size, opt_gpu):
  """Generates CNN features using PyTorch"""


  import os
  from os.path import join
  from pathlib import Path

  import cv2 as cv
  import numpy as np
  from PIL import Image
  from tqdm import tqdm

  from vframe.settings.paths import Paths
  from vframe.utils import file_utils, im_utils, logger_utils
  from vframe.models.metadata_item import FeatureMetadataItem
  from vframe.processors.feature_extractor_pytorch import FeatureExtractor
  
  # -------------------------------------------------
  # process 

  log = logger_utils.Logger.getLogger()

  # select type of metadata
  if opt_net == types.PyTorchNet.RESNET18:
    metadata_type = types.Metadata.FEATURE_PT_RESNET18
  elif opt_net == types.PyTorchNet.ALEXNET:
    metadata_type = types.Metadata.FEATURE_PT_ALEXNET
  
  log.debug('PyTorch feature vectors using: {}'.format(metadata_type.name.lower()))

  dir_media = Paths.media_dir(types.Metadata.KEYFRAME, 
    data_store=opt_disk, verified=ctx.opts['verified'])
  opt_size_label = cfg.IMAGE_SIZE_LABELS[opt_size]

  # initialize feature extractor
  fe = FeatureExtractor(cuda=(opt_gpu > -1), net=opt_net)

  # iterate process images
  while True:
  
    chair_item = yield
    
    media_record = chair_item.media_record
    sha256 = media_record.sha256
    sha256_tree = file_utils.sha256_tree(sha256)
    dir_sha256 = join(dir_media, sha256_tree, sha256)

    # get the keyframe status data to check if images available
    try:
      keyframe_status = media_record.get_metadata(types.Metadata.KEYFRAME_STATUS)
    except Exception as ex:
      log.error('no keyframe metadata. Try: "append -t keyframe_status"')
      return

    # if keyframe images were generated and exist locally
    metadata = {}
    try:
      status = keyframe_status.get_status(opt_size)
    except:
      status = False

    if keyframe_status and keyframe_status.get_status(opt_size):
      try:
        keyframe_metadata = media_record.get_metadata(types.Metadata.KEYFRAME)
      except Exception as ex:
        log.error('no keyframe metadata. Try: "append -t keyframe"')
        return

      # get keyframe indices
      idxs = keyframe_metadata.get_keyframes(opt_density)

      # generate metadata
      for idx in idxs:
        # get keyframe filepath
        fp_keyframe = join(dir_sha256, file_utils.zpad(idx), opt_size_label, 'index.jpg')
      try:
        im = cv.imread(fp_keyframe)
        vec = fe.extract(im)
        metadata[idx] = vec.tolist()  # convert to list, JSON safe
      except Exception as ex:
        log.error('Exception: {}'.format(ex))
        if not Path(fp_keyframe).exists():
          log.error('file not found: {}'.format(fp_keyframe))
        else:
          log.error('could not compute features: {}'.format(fp_keyframe))
        metadata[idx] = []  # set to empty if error
    
    # append metadata to chair_item's mapping item
    chair_item.media_record.set_metadata(metadata_type, FeatureMetadataItem(metadata))


    # -------------------------------------------------   
    # send back to generator
    sink.send(chair_item)

