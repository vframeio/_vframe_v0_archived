"""
Generate metadata using OpenCV's DNN module
- under development
- about 10FPS? on i7 CPU 12x
- using Python Yolo is much faster w/GPU
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
  default=click_utils.get_default(types.KeyframeMetadata.EXPANDED),
  show_default=True,
  type=cfg.KeyframeMetadataVar,
  help=click_utils.show_help(types.KeyframeMetadata))
@click.option('--size', 'opt_size',
  type=cfg.ImageSizeVar,
  default=click_utils.get_default(types.ImageSize.MEDIUM),
  help=click_utils.show_help(types.ImageSize))
@click.option('--display/--no-display', 'opt_display', is_flag=True,
  help='Display the image')
@processor
@click.pass_context
def cli(ctx, sink, opt_disk, opt_density, opt_size, opt_display):
  """Generates detections with CV DNN"""

  # ----------------------------------------------------------------
  # imports

  import os
  from os.path import join
  from pathlib import Path

  import click
  import cv2 as cv
  import numpy as np

  from vframe.utils import click_utils, file_utils, im_utils, logger_utils
  from vframe.models.metadata_item import DetectMetadataItem, DetectResult
  from vframe.settings.paths import Paths

  # ----------------------------------------------------------------
  # init

  log = logger_utils.Logger.getLogger()

  metadata_type = types.Metadata.COCO

  # directory for images
  dir_media = Paths.media_dir(types.Metadata.KEYFRAME, data_store=opt_disk, verified=ctx.opts['verified'])
  opt_size_label = cfg.IMAGE_SIZE_LABELS[opt_size]

  # Initialize the parameters


  # TODO externalize function

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

  # TODO externalize to Paths    
  DIR_DARKNET = join(cfg.DIR_MODELS, 'darknet/pjreddie')
  fp_weights = join(DIR_DARKNET, 'weights/yolov3.weights')
  fp_cfg = join(DIR_DARKNET, 'cfg/yolov3.cfg')
  fp_classes = join(DIR_DARKNET, 'data/coco.names')

  classes = file_utils.load_text(fp_classes)
  net = cv.dnn.readNetFromDarknet(fp_cfg, fp_weights)
  net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
  net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

  # TODO sort this out

  # # network input w, h
  nms_thresh = 0.45
  hier_thresh = 0.5
  conf_thresh = 0.5

  # initialize dnn
  dnn_size = (416, 416)  # network input w, h
  dnn_clr = (0, 0, 0)  # mean color to subtract
  dnn_scale = 1/255
  dnn_threshold = 0.85  #Confidence threshold
  nms_threshold = 0.4   #Non-maximum suppression threshold
  dnn_px_range = 1  # pixel value range
  dnn_crop = False

  # ----------------------------------------------------------------
  # process

  # iterate sink
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
      # TODO make exception/error logging class for this error
      log.error('no keyframe metadata. Try: "append -t keyframe_status"')
      return

    # if keyframe images were generated and exist locally
    metadata = {}
    if keyframe_status and keyframe_status.get_status(opt_size):
      try:
        keyframe_metadata = media_record.get_metadata(types.Metadata.KEYFRAME)
      except Exception as ex:
        # TODO make exception/error logging class for this error
        log.error('no keyframe metadata. Try: "append -t keyframe"')
        return

      # get keyframe indices
      idxs = keyframe_metadata.get_keyframes(opt_density)

      for frame_idx in idxs:
        # get keyframe filepath
        fp_keyframe = join(dir_sha256, file_utils.zpad(frame_idx), opt_size_label, 'index.jpg')
        # Create a 4D blob from a frame.
        im = cv.imread(fp_keyframe)
        blob = cv.dnn.blobFromImage(im, dnn_scale, dnn_size, dnn_clr, dnn_px_range, crop=dnn_crop)
        # Sets the input to the network
        net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        net_outputs = net.forward(getOutputsNames(net))

        det_results = post_process(im, net_outputs, dnn_threshold, nms_threshold)
        metadata[frame_idx] = det_results
    
    # append metadata to chair_item's mapping item
    chair_item.item.set_metadata(metadata_type, DetectMetadataItem(metadata))
  
    # ----------------------------------------------------------------
    # yield back to the processor pipeline

    # send back to generator
    sink.send(chair_item)



# ----------------------------------------------------------------
# little helpers

# NMS post processing functions by @spmallick 
# https://github.com/spmallick/learnopencv/tree/master/ObjectDetection-YOLO
# https://www.learnopencv.com/deep-learning-based-object-detection-using-yolov3-with-opencv-python-c/


# Get the names of the output layers
def getOutputsNames(net):
  # Get the names of all the layers in the network
  layersNames = net.getLayerNames()
  # Get the names of the output layers, i.e. the layers with unconnected outputs
  return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def post_process(im, net_outputs, dnn_threshold, nms_threshold):
  im_h, im_w = im.shape[:2]

  # Scan through all the bounding boxes output from the network and keep only the
  # ones with high confidence scores. Assign the box's class label as the class with the highest score.
  class_ids = []
  confidences = []
  boxes = []
  for net_output in net_outputs:
    for detection in net_output:
      scores = detection[5:]
      class_id = np.argmax(scores)
      confidence = scores[class_id]
      if confidence > dnn_threshold:
        cx = int(detection[0] * im_w)
        cy = int(detection[1] * im_h)
        w = int(detection[2] * im_w)
        h = int(detection[3] * im_h)
        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        class_ids.append(class_id)
        confidences.append(float(confidence))
        boxes.append([x1, y1, w, h])

  # Perform non maximum suppression to eliminate redundant overlapping boxes with
  nms_objs = cv.dnn.NMSBoxes(boxes, confidences, dnn_threshold, nms_threshold)

  results = []
  for nms_obj in nms_objs:
    nms_idx = nms_obj[0]
    class_idx = class_ids[nms_idx]
    score = confidences[nms_idx]
    bbox = boxes[nms_idx]
    results.append(DetectResult(class_idx, score, bbox))

  return results
