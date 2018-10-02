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


@click.command('gen_darknet_coco')
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
@click.option('-t', '--net-type', 'opt_net',
  type=cfg.DarknetDetectVar,
  default=click_utils.get_default(types.DarknetDetect.SUBMUNITION),
  help=click_utils.show_help(types.DarknetDetect))
@click.option('--display/--no-display', 'opt_display', is_flag=True, 
  help='Display the image')
@click.option('--delay', 'opt_delay', default=250,
  help='Millisecond delay between images if display')
@click.option('-g', '--gpu', 'opt_gpu', default=0,
  help='GPU index')
@processor
@click.pass_context
def cli(ctx, sink, opt_disk, opt_density, opt_size, opt_net, opt_display, 
  opt_delay, opt_gpu):
  """Generates detections with Darknet"""

  
  # -------------------------------------------------
  # imports 

  import os
  from os.path import join
  from pathlib import Path
  from io import BytesIO

  import PIL.Image
  import numpy as np
  import requests
  import cv2 as cv
  import numpy as np
  # temporary fix, update pydarknet to >= 0.1rc12
  # pip install yolo34py-gpu==0.1rc13
  os.environ['CUDA_VISIBLE_DEVICES'] = str(opt_gpu)
  import pydarknet
  from pydarknet import Detector
  from pydarknet import Image as DarknetImage

  from vframe.utils import file_utils, im_utils, logger_utils
  from vframe.models.metadata_item import DetectMetadataItem, DetectResult
  from vframe.settings.paths import Paths


  
  # -------------------------------------------------
  # initialize

  log = logger_utils.Logger.getLogger()


  # process keyframes
  dir_media = Paths.media_dir(types.Metadata.KEYFRAME, data_store=opt_disk, 
    verified=ctx.opts['verified'])
  opt_size_label = cfg.IMAGE_SIZE_LABELS[opt_size]

  # Initialize the parameters
  fp_cfg = Paths.darknet_cfg(opt_net=opt_net, data_store=opt_disk)
  fp_weights = Paths.darknet_weights(opt_net=opt_net, data_store=opt_disk)
  fp_data = Paths.darknet_data(opt_net=opt_net, data_store=opt_disk)
  fp_classes = Paths.darknet_classes(opt_net=opt_net, data_store=opt_disk)
  log.debug('fp_classes: {}'.format(fp_classes))
  class_names = file_utils.load_text(fp_classes)
  class_idx_lookup = {label: i for i, label in enumerate(class_names)}

  # init Darknet detector
  # pydarknet.set_cuda_device(opt_gpu)  # not yet implemented in 0.1rc12
  net = Detector(fp_cfg, fp_weights, 0, fp_data)

  # # network input w, h
  nms_thresh = 0.45
  hier_thresh = 0.5
  conf_thresh = 0.5


  # Use mulitples of 32: 416, 448, 480, 512, 544, 576, 608, 640, 672, 704
  if opt_net == types.DarknetDetect.OPENIMAGES:
    metadata_type = types.Metadata.OPENIMAGES
    dnn_size = (608, 608)
    dnn_threshold = 0.875
  elif  opt_net == types.DarknetDetect.COCO:
    metadata_type = types.Metadata.COCO
    dnn_size = (416, 416)
    dnn_threshold = 0.925
  elif  opt_net == types.DarknetDetect.COCO_SPP:
    metadata_type = types.Metadata.COCO
    dnn_size = (608, 608)
    dnn_threshold = 0.875
  elif  opt_net == types.DarknetDetect.VOC:
    metadata_type = types.Metadata.VOC
    dnn_size = (416, 416)
    dnn_threshold = 0.875
  elif  opt_net == types.DarknetDetect.SUBMUNITION:
    metadata_type = types.Metadata.SUBMUNITION
    dnn_size = (608, 608)
    dnn_threshold = 0.90

    # TODO: function to collapse hierarchical detections into parent class
    # flatten hierarchical objects
    # ao25_idxs = [0, 1, 2, 3, 7, 8, 9, 11]
    # ao25_idx = 0
    # shoab_idxs = [6, 10]
    # shoab_idx = 12
    # cassette_idxs = [4, 5]
    # cassette_idx = 5

    # for k, v in class_idx_lookup.copy().items():
    #   if v in ao25_idxs:
    #     v = ao25_idx
    #   elif v in shoab_idxs:
    #     v = shoab_idx
    #   elif v in cassette_idxs:
    #     v = cassette_idx
    #   class_idx_lookup[k] = v  


  # -------------------------------------------------
  # process 
  
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
    if keyframe_status and keyframe_status.get_status(opt_size):
      
      keyframe_metadata = media_record.get_metadata(types.Metadata.KEYFRAME)
      
      if not keyframe_metadata:
        log.error('no keyframe metadata. Try: "append -t keyframe"')
        return

      # get keyframe indices
      idxs = keyframe_metadata.get_keyframes(opt_density)

      for frame_idx in idxs:
        # get keyframe filepath
        fp_keyframe = join(dir_sha256, file_utils.zpad(frame_idx), opt_size_label, 'index.jpg')
        try:
          im = cv.imread(fp_keyframe)
          im.shape  # used to invoke error if file didn't load correctly
        except:
          log.warn('file not found: {}'.format(fp_keyframe))
          continue

        # -------------------------------------------
        # Start DNN
        im_sm = im_utils.resize(im, width=dnn_size[0], height=dnn_size[1])
        imh, imw = im_sm.shape[:2]
        im_dk = DarknetImage(im_sm)
        net_outputs = net.detect(im_dk, thresh=conf_thresh, hier_thresh=hier_thresh, nms=nms_thresh)
        # threshold
        net_outputs = [x for x in  net_outputs if float(x[1]) > dnn_threshold]

        # append as metadata
        det_results = []
        for cat, score, bounds in net_outputs:
          cx, cy, w, h = bounds
          # TODO convert to BBox()
          x1, y1 = ( int(max(cx - w / 2, 0)), int(max(cy - h / 2, 0)) )
          x2, y2 = ( int(min(cx + w / 2, imw)), int(min(cy + h / 2, imh)) )
          class_idx = class_idx_lookup[cat.decode("utf-8")]
          rect_norm = (x1/imw, y1/imh, x2/imw, y2/imh)
          det_results.append( DetectResult(class_idx, float(score), rect_norm) )
          
        # display to screen
        # TODO: replace this with drawing functions
        if opt_display and len(net_outputs) > 1:
          
          im_dst = im_sm.copy()
          for cat, score, bounds in net_outputs:
            cx, cy, w, h = bounds
            # TODO convert to BBox()
            x1, y1 = ( int(max(cx - w / 2, 0)), int(max(cy - h / 2, 0)) )
            x2, y2 = ( int(min(cx + w / 2, imw)), int(min(cy + h / 2, imh)) )
            # TODO convert to drawing processor
            cv.rectangle(im_dst, (x1, y1), (x2, y2) , (0, 255, 0), thickness=2)
            label = str(cat.decode("utf-8"))
            label_idx = class_idx_lookup[label]
            label = '{} ({:.2f})'.format(class_names[label_idx].upper(), float(score))
            twh = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, .4, 1)
            cv.rectangle(im_dst, (x1, y1), (x1+twh[0][0]+5, y1+twh[0][1]+5) , (0, 255, 0), -1)
            cv.putText(im_dst, label, (int(x1+2),int(y1+13)),cv.FONT_HERSHEY_SIMPLEX,.4,(0,0,0), 1)

          cv.imshow('frame'.format(frame_idx), im_dst)
          k = cv.waitKey(opt_delay)
          if k == 27 or k == ord('q'):  # ESC
            # exits the app
            cv.destroyAllWindows()
            return

        metadata[frame_idx] = det_results
    
    # append metadata to chair_item's mapping item
    chair_item.item.set_metadata(metadata_type, DetectMetadataItem(metadata))
  

    # send back to generator
    sink.send(chair_item)
