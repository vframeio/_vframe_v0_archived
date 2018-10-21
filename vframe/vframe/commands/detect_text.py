"""
Scene text detection using OpenCV DNN
https://github.com/opencv/opencv_contrib/tree/master/modules/text
https://github.com/opencv/opencv_contrib/blob/master/modules/text/samples/textdetection.py

DeepScene Text Detector
A demo script of text box alogorithm of the paper
Minghui Liao et al.: TextBoxes: A Fast Text Detector with a Single Deep Neural Network
https://arxiv.org/abs/1611.06779

EAST Text Detector
https://bitbucket.org/tomhoag/opencv-text-detection/overview
https://github.com/argman/EAST

TODO
- add angled text detector

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
@click.option('--conf', 'opt_conf_thresh', 
  type=click.FloatRange(0,1), default=0.95,
  help='Minimum detection confidence')
@click.option('--nms', 'opt_nms_thresh',
  type=click.FloatRange(0,1), default=0.4,
  help='Minimum non-max supression confidence')
@click.option('-t', '--net-type', 'opt_net',
  type=cfg.SceneTextNetVar,
  default=click_utils.get_default(types.SceneTextNet.EAST),
  help=click_utils.show_help(types.SceneTextNet))
@processor
@click.pass_context
def cli(ctx, sink, opt_disk, opt_net, opt_conf_thresh, opt_nms_thresh):
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
  from vframe.models.metadata_item import ROIMetadataItem, ROIDetectResult
  from vframe.settings.paths import Paths
  from vframe.models.bbox import BBox

  # ----------------------------------------------------------------
  # init

  log = logger_utils.Logger.getLogger()


  metadata_type = types.Metadata.TEXT_ROI
  
  # initialize dnn
  if opt_net == types.SceneTextNet.EAST:
    # TODO externalize
    dnn_size = (320, 320)  # fixed
    dnn_mean_clr = (123.68, 116.78, 103.94)  # fixed
    dnn_scale = 1.0  # fixed
    dnn_layer_names = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
    fp_model = join(cfg.DIR_MODELS_TF, 'east', 'frozen_east_text_detection.pb')
    log.debug('fp_model: {}'.format(fp_model))
    net = cv.dnn.readNet(fp_model)
    #net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    #net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

  elif  opt_net == types.SceneTextNet.DEEPSCENE:
    dnn_size = (320, 320)  # fixed
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

      if opt_net == types.SceneTextNet.DEEPSCENE:
        # DeepScene scene text detector (opencv contrib)
        frame = im_utils.resize(frame, width=dnn_size[0], height=dnn_size[1])
        frame_dim = frame.shape[:2][::-1]
        rects, probs = net.detect(frame)
        det_results = []
        for r in range(np.shape(rects)[0]):
          prob = float(probs[r])
          if prob > opt_conf_thresh:
            rect = BBox.from_xywh_dim(*rects[r], frame_dim).as_xyxy()  # normalized
            det_results.append( ROIDetectResult(prob, rect))
              
        metadata[frame_idx] = det_results

      elif types.SceneTextNet.EAST:
        # EAST scene text detector
        frame = im_utils.resize(frame, width=dnn_size[0], height=dnn_size[1])
        # frame = im_utils.resize(frame, width=dnn_size[0], he)
        frame_dim = frame.shape[:2][::-1]
        frame_dim = dnn_size

        # blob = cv.dnn.blobFromImage(frame, dnn_scale, dnn_size, dnn_mean_clr, swapRB=True, crop=False)
        blob = cv.dnn.blobFromImage(frame, dnn_scale, dnn_size, dnn_mean_clr, swapRB=True, crop=False)
        net.setInput(blob)
        (scores, geometry) = net.forward(dnn_layer_names)
        (rects, confidences, baggage) = scenetext_utils.east_text_decode(scores, geometry, opt_conf_thresh)
        
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
            confidence_threshold=opt_conf_thresh,
            nsm_threshold=opt_nms_thresh)
        
          indicies = np.array(indicies).reshape(-1)
          rects = np.array(rects)[indicies]
          scores = np.array(confidences)[indicies]
          for rect, score in zip(rects, scores):
            rect = BBox.from_xywh_dim(*rect, frame_dim).as_xyxy()  # normalized
            det_results.append( ROIDetectResult(score, rect) )

        metadata[frame_idx] = det_results

        
    # append metadata to chair_item's mapping item
    chair_item.set_metadata(metadata_type, ROIMetadataItem(metadata))
  
    # ----------------------------------------------------------------
    # yield back to the processor pipeline

    # send back to generator
    sink.send(chair_item)


