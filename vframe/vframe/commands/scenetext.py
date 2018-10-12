"""
Scene text detection using OpenCV DNN
https://github.com/opencv/opencv_contrib/tree/master/modules/text
https://github.com/opencv/opencv_contrib/blob/master/modules/text/samples/textdetection.py

DeepScene Text Detector
A demo script of text box alogorithm of the paper
Minghui Liao et al.: TextBoxes: A Fast Text Detector with a Single Deep Neural Network
https://arxiv.org/abs/1611.06779

EAST Text Detector

"""

import click

from vframe.settings import types
from vframe.utils import click_utils
from vframe.settings import vframe_cfg as cfg

from cli_vframe import processor

@click.command()
@click.option('-d', '--disk', 'opt_disk',
  default=click_utils.get_default(types.DataStore.HDD),
  type=cfg.DataStoreVar,
  show_default=True,
  help=click_utils.show_help(types.DataStore))
@click.option('-t', '--net-type', 'opt_net',
  type=cfg.SceneTextNetVar,
  default=click_utils.get_default(types.SceneTextNet.DEEPSCENE),
  help=click_utils.show_help(types.SceneTextNet))
@processor
@click.pass_context
def cli(ctx, sink, opt_disk, opt_net):
  """Generates scene text ROIs (CV DNN)"""

  # ----------------------------------------------------------------
  # imports

  import os
  from os.path import join
  from pathlib import Path

  import click
  import cv2 as cv
  import numpy as np
  from nms import nms

  from vframe.utils import click_utils, file_utils, im_utils, logger_utils, dnn_utils
  from vframe.utils import scenetext_utils
  from vframe.models.metadata_item import SceneTextDetectMetadataItem, SceneTextDetectResult
  from vframe.settings.paths import Paths
  from vframe.models.bbox import BBox

  # ----------------------------------------------------------------
  # init

  log = logger_utils.Logger.getLogger()


  # TODO externalize function

  # initialize dnn
  dnn_clr = (0, 0, 0)  # mean color to subtract
  dnn_scale = 1/255  # ?
  nms_threshold = 0.4   #Non-maximum suppression threshold
  dnn_px_range = 1  # pixel value range ?
  dnn_crop = False  # probably crop or force resize

  # Use mulitples of 32: 416, 448, 480, 512, 544, 576, 608, 640, 672, 704
  if opt_net == types.SceneTextNet.EAST:
    metadata_type = types.Metadata.TEXTROI
    dnn_size = (320, 320)
    dnn_mean_clr = (123.68, 116.78, 103.94)
    dnn_scale = 1.0
    nms_thresh = 0.4
    conf_thresh = 0.5
    dnn_layer_names = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
    fp_model = join(cfg.DIR_MODELS_TF, 'east', 'frozen_east_text_detection.pb')
    log.debug('fp_model: {}'.format(fp_model))
    net = cv.dnn.readNet(fp_model)
    # net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    # net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

  elif  opt_net == types.SceneTextNet.DEEPSCENE:
    metadata_type = types.Metadata.TEXTROI
    dnn_size = (320, 320)
    conf_thresh = 0.5
    fp_model = join(cfg.DIR_MODELS_CAFFE, 'deepscenetext', "TextBoxes_icdar13.caffemodel")
    fp_prototxt = join(cfg.DIR_MODELS_CAFFE, 'deepscenetext', 'textbox.prototxt')
    net = cv.text.TextDetectorCNN_create(fp_prototxt, fp_model)


  # ----------------------------------------------------------------
  # process

  # iterate sink
  while True:
    
    chair_item = yield
    
    metadata = {}
    
    for frame_idx, frame in chair_item.keyframes.items():

      # detect
      imh, imw = frame.shape[:2]

      if opt_net == types.SceneTextNet.DEEPSCENE:
        
        frame = im_utils.resize(frame, width=160)
        rects, probs = net.detect(frame)
        det_results = []
        for r in range(np.shape(rects)[0]):
          prob = float(probs[r])
          if prob > conf_thresh:
            log.debug('thresh ok: {}'.format(prob))
            x1, y1, w, h = rects[r]
            log.debug('make bbox')
            rect_norm = BBox(x1, y1, x1 + w, y1 + h, imw, imh).as_norm()
            log.debug('append scene text')
            det_results.append( SceneTextDetectResult('hello', prob, rect_norm) )
            log.debug('found text: {}'.format(rect_norm))
              
        metadata[frame_idx] = det_results

      elif types.SceneTextNet.EAST:

        frame = im_utils.resize(frame, width=dnn_size[0], height=dnn_size[1])
        blob = cv.dnn.blobFromImage(frame, dnn_scale, dnn_size, dnn_mean_clr, swapRB=True, crop=False)
        net.setInput(blob)
        (scores, geometry) = net.forward(dnn_layer_names)
        (rects, confidences, baggage) = scenetext_utils.east_text_decode(scores, geometry, conf_thresh)
        det_results = []
        if rects:
          offsets = []
          thetas = []
          for b in baggage:
            offsets.append(b['offset'])
            thetas.append(b['angle'])

          # functions = [nms.felzenszwalb.nms, nms.fast.nms, nms.malisiewicz.nms]
          indicies = nms.boxes(rects, confidences, 
            nms_function=nms.fast.nms, 
            confidence_threshold=conf_thresh,
            nsm_threshold=nms_thresh)
        
          indicies = np.array(indicies).reshape(-1)
          rects = np.array(rects)[indicies]
          scores = np.array(confidences)[indicies]
          for rect, score in zip(rects, scores):
            rect_norm = BBox.from_xywh(rect, dnn_size[0], dnn_size[1]).as_norm()
            det_results.append( SceneTextDetectResult('EAST', score, rect_norm) )

        metadata[frame_idx] = det_results

        
    # append metadata to chair_item's mapping item
    chair_item.set_metadata(metadata_type, SceneTextDetectMetadataItem(metadata))
  
    # ----------------------------------------------------------------
    # yield back to the processor pipeline

    # send back to generator
    sink.send(chair_item)


