"""
Generates metadata using Yolo/Darknet Python interface
- about 20-30 FPS on NVIDIA 1080 Ti GPU
- SPP currently not working
- enusre image size matches network image size

"""
import click

from vframe.settings import types
from vframe.utils import click_utils
from vframe.settings import vframe_cfg as cfg

from cli_vframe import processor


@click.command('')
@click.option('--dir-media', 'opt_dir_media', default=None,
  help='Path to media folder')
@click.option('-d', '--disk', 'opt_disk',
  default=click_utils.get_default(types.DataStore.SSD),
  type=cfg.DataStoreVar,
  show_default=True,
  help=click_utils.show_help(types.DataStore))
@click.option('--density', 'opt_density',
  default=click_utils.get_default(types.KeyframeMetadata.BASIC),
  show_default=True,
  type=cfg.KeyframeMetadataVar,
  help=click_utils.show_help(types.KeyframeMetadata))
@click.option('--size', 'opt_size_type',
  type=cfg.ImageSizeVar,
  default=click_utils.get_default(types.ImageSize.MEDIUM),
  help=click_utils.show_help(types.ImageSize))
@click.option('--draw', 'opt_drawframes', is_flag=True, default=False,
  help='Add drawframes if drawing')
@processor
@click.pass_context
def cli(ctx, sink, opt_dir_media, opt_disk, opt_density, opt_size_type, opt_drawframes):
  """Appends images to ChairItem"""
  
  # -------------------------------------------------
  # imports 

  from os.path import join

  import cv2 as cv
  
  from vframe.utils import file_utils, logger_utils
  from vframe.settings.paths import Paths

  
  # -------------------------------------------------
  # initialize

  log = logger_utils.Logger.getLogger()
  log.debug('append images to pipeline')

  # process keyframes
  if not opt_dir_media:
    dir_media = Paths.media_dir(types.Metadata.KEYFRAME, data_store=opt_disk, 
      verified=ctx.opts['verified'])
  else:
    dir_media = opt_dir_media

  
  # -------------------------------------------------
  # process 
  
  while True:
    
    chair_item = yield
    
    if chair_item.chair_type == types.ChairItemType.PHOTO:
      chair_item.load_images(dir_media, opt_size_type, opt_drawframes=opt_drawframes)
    if chair_item.chair_type == types.ChairItemType.VIDEO:
      pass
      #chair_item.load_images(opt_size_type, opt_drawframes=opt_drawframes)
    if chair_item.chair_type == types.ChairItemType.VIDEO_KEYFRAME:
      chair_item.load_images(opt_size_type, opt_drawframes=opt_drawframes)
    if chair_item.chair_type == types.ChairItemType.MEDIA_RECORD:
      chair_item.load_images(dir_media, opt_size_type, opt_density, opt_drawframes=opt_drawframes)
    # ------------------------------------------------------------
    # send back to generator

    sink.send(chair_item)
