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
@click.option('-a', '--action', 'opt_action', required=True,
  type=cfg.ActionVar,
  default=click_utils.get_default(types.Action.ADD),
  help=click_utils.show_help(types.Action))
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
@click.option('--size', 'opt_size',
  type=cfg.ImageSizeVar,
  default=click_utils.get_default(types.ImageSize.MEDIUM),
  help=click_utils.show_help(types.ImageSize))
@click.option('--draw', 'opt_drawframes', is_flag=True, default=False,
  help='Add drawframes if drawing')
@processor
@click.pass_context
def cli(ctx, sink, opt_action, opt_dir_media, opt_disk, opt_density, opt_size, opt_drawframes):
  """Loads keyframe images"""

  
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

  opt_size_label = cfg.IMAGE_SIZE_LABELS[opt_size]

  
  # -------------------------------------------------
  # process 
  
  while True:
    
    chair_item = yield
    
    if opt_action == types.Action.ADD:

      if chair_item.chair_type == types.ChairItemType.VIDEO:
        log.debug('load_video_keyframes start')
        chair_item.load_video_keyframes(opt_drawframes=opt_drawframes)
        log.debug('load_video_keyframes done')
      elif chair_item.chair_type == types.ChairItemType.MEDIA_RECORD:
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

          keyframes = {}

          # if keyframe images were generated and exist locally
          if keyframe_status and keyframe_status.get_status(opt_size):
            
            keyframe_metadata = media_record.get_metadata(types.Metadata.KEYFRAME)
            
            if not keyframe_metadata:
              log.error('no keyframe metadata. Try: "append -t keyframe"')
              return

            # get keyframe indices
            frame_idxs = keyframe_metadata.get_keyframes(opt_density)

            for frame_idx in frame_idxs:
              # get keyframe filepath
              fp_keyframe = join(dir_sha256, file_utils.zpad(frame_idx), opt_size_label, 'index.jpg')
              try:
                im = cv.imread(fp_keyframe)
                im.shape  # used to invoke error if file didn't load correctly
              except:
                log.warn('file not found: {}'.format(fp_keyframe))
                # don't add to keyframe dict
                continue

              keyframes[frame_idx] = im
          
          
          # append metadata to chair_item's mapping item
          chair_item.set_keyframes(keyframes, opt_drawframes)

    elif opt_action == types.Action.RM:

      chair_item.remove_keyframes()
      chair_item.remove_drawframes()
    

    # ------------------------------------------------------------
    # send back to generator

    sink.send(chair_item)
