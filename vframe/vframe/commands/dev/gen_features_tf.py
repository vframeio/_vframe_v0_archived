import click
from tqdm import tqdm

from vframe.settings import types
from vframe.utils import click_utils

from cli_vframe import processor

import os
from os.path import join
from pathlib import Path

import cv2 as cv
import numpy as np

from vframe.utils import file_utils, im_utils, logger_utils
from vframe.models.metadata_item import FeatureMetadataItem
from vframe.settings.paths import Paths
from vframe.settings import vframe_cfg as cfg

import tensorflow as tf
import keras.backend as K
from keras.backend.tensorflow_backend import set_session


fe = None  # feature extractor global var, can probably remove

@click.command('feature_extractor')
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
  default=click_utils.get_default(types.ImageSize.MEDIUM),
  help=click_utils.show_help(types.ImageSize))
@click.option('--net', 'opt_net',
  default=click_utils.get_default(types.KerasNet.VGG16),
  type=click_utils.KerasNetVar,
  help=click_utils.show_help(types.KerasNet))
@click.option('--weights', 'opt_weights', default='imagenet',type=str,
  help='Path to kearas weights or use default imagenet')
@click.option('--gpu', 'opt_gpu', type=int, default=0,
  help='GPU index (use -1 for CPU)')
@click.option('--ram', 'opt_ram', default=30,
  type=click.IntRange(30, 100, clamp=True),
  help='GPU RAM limt percentage')
@processor
@click.pass_context
def cli(ctx, sink, opt_disk, opt_density, opt_size, opt_net, opt_weights, opt_gpu, opt_ram):
  """Checks if keyframe images exist"""

  # -------------------------------------------------
  # imports 

  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
  # this needs to be fixed
  # causes issues when imports placed inside cli function
  # possibly related to sessions and click processors
  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  # load tf first to limit GPU RAM growth
  #   needs to be in this file (caused issues when used in import file)
  #   https://github.com/keras-team/keras/issues/4161#issuecomment-366031228
  
  # working as of sept 27, but may still be buggy


  config = tf.ConfigProto()
  config.gpu_options.per_process_gpu_memory_fraction = opt_ram / 100
  config.gpu_options.visible_device_list = str(opt_gpu)
  set_session(tf.Session(config=config))
  # then import keras ?!
  import keras.applications
  from keras.models import Model  

  from vframe.processors.feature_extractor_tf import FeatureExtractor
  
  # -------------------------------------------------
  # process 

  log = logger_utils.Logger.getLogger()
  log.debug('Keras CNN feature vectors')

  metadata_type = types.Metadata.FEATURE_VGG16

  dir_media = Paths.media_dir(types.Metadata.KEYFRAME, data_store=opt_disk, verified=ctx.opts['verified'])
  opt_size_label = cfg.IMAGE_SIZE_LABELS[opt_size]

  # deprecated
  fe = FeatureExtractor(net=opt_net, weights=opt_weights)

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
          vec = fe.extract_fp(fp_keyframe)
          metadata[idx] = vec.tolist()  # convert to list, JSON safe
        except:
          if not Path(fp_keyframe).exists():
            log.error('file not found: {}'.format(fp_keyframe))
          else:
            log.error('could not compute features: {}'.format(fp_keyframe))
          metadata[idx] = []
    
    # append metadata to chair_item's mapping item
    chair_item.media_record.set_metadata(metadata_type, FeatureMetadataItem(metadata))


    # -------------------------------------------------   
    # send back to generator
    sink.send(chair_item)

