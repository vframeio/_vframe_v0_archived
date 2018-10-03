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
@click.option('-t', '--net-type', 'opt_net',
  default=click_utils.get_default(types.PyTorchNet.RESNET18),
  type=cfg.PyTorchNetVar,
  help=click_utils.show_help(types.PyTorchNet))
@click.option('--gpu', 'opt_gpu', type=int, default=0,
  help='GPU index (use -1 for CPU)')
@processor
@click.pass_context
def cli(ctx, sink, opt_disk, opt_net, opt_gpu):
  """Generates CNN features using PyTorch"""

  # -------------------------------------------------
  # imports

  from vframe.utils import logger_utils, im_utils
  from vframe.models.metadata_item import FeatureMetadataItem
  from vframe.processors.feature_extractor import FeatureExtractor
  

  # -------------------------------------------------
  # init

  # convert feature network type to metadata type
  if opt_net == types.PyTorchNet.RESNET18:
    metadata_type = types.Metadata.FEATURE_RESNET18
  elif opt_net == types.PyTorchNet.ALEXNET:
    metadata_type = types.Metadata.FEATURE_ALEXNET
  
  log = logger_utils.Logger.getLogger()
  log.debug('PyTorch feature vectors using: {}'.format(metadata_type.name.lower()))

  # initialize feature extractor
  fe = FeatureExtractor(cuda=(opt_gpu > -1), opt_net=opt_net)

  # -------------------------------------------------
  # process

  while True:
  
    chair_item = yield
    
    # check if no images
    if not len(chair_item.keyframes.keys()) > 0:
      log.warn('no images for {}'.format(chair_item.sha256))  #  try adding "images" to command?

    metadata = {}
    # iterate keyframes and extract feature vectors as serialized data
    for frame_idx, frame in chair_item.keyframes.items():
      frame_pil = im_utils.ensure_pil(frame, bgr2rgb=True)
      metadata[frame_idx] = fe.extract(frame_pil, to_list=True)
    
    # append metadata to chair_item's mapping item
    chair_item.media_record.set_metadata(metadata_type, FeatureMetadataItem(metadata))

    # -------------------------------------------------   
    # send back to generator
    
    sink.send(chair_item)

