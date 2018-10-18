"""
Face detection: detects face ROI with options for different algorithms

DLIB CNN:
- 60-80 FPS with GTX 1080 Ti on i7x12
- 100-200 FPS batch process up to 100 items (128 images creates OOM error @11GB RAM)
- accuracy: high

DLIB HOG:
- 35-40 FPS with GTX 1080 Ti on i7x12 
- accuracy: medium

OpenCV DNN:
- 65-75 FPS with i7x12
- accuracy: high

OpenCV HAAR:
- FPS: not yet implemented 
- accuracy: low

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
  type=cfg.FaceDetectNetVar,
  default=click_utils.get_default(types.FaceDetectNet.DLIB_CNN),
  help=click_utils.show_help(types.FaceDetectNet))
@click.option('--size', 'opt_dnn_size', 
  type=(int, int), default=(500, 500),
  help='Inference image size. Default: 300x300 OpenCV DNN')
@click.option('--pyramids', 'opt_pyramids', 
  default=2,
  help='Number of image pyramids for dlib')
@click.option('--conf', 'opt_conf_thresh', 
  type=click.FloatRange(0,1), default=0.5,  # 0.875 for CVDNN, 0.5 for DLIB
  help='Minimum detection confidence')
@click.option('-g', '--gpu', 'opt_gpu', default=0,
  help='GPU index')
@processor
@click.pass_context
def cli(ctx, sink, opt_disk, opt_net, opt_dnn_size, opt_pyramids, opt_conf_thresh, opt_gpu):
  """Generates face detection ROIs"""

  # ----------------------------------------------------------------
  # imports

  import os
  from os.path import join
  from pathlib import Path

  import click
  import cv2 as cv
  import dlib
  import numpy as np
  from nms import nms

  from vframe.utils import click_utils, file_utils, im_utils, logger_utils, dnn_utils
  from vframe.models.metadata_item import ROIMetadataItem, ROIDetectResult
  from vframe.settings.paths import Paths
  from vframe.models.bbox import BBox

  # ----------------------------------------------------------------
  # init

  log = logger_utils.Logger.getLogger()

  metadata_type = types.Metadata.FACE_ROI

  if opt_net == types.FaceDetectNet.CVDNN:
    dnn_scale = 1.0  # fixed
    dnn_mean = (104.0, 177.0, 123.0)  # fixed
    dnn_crop = False  # probably crop or force resize
    fp_prototxt = join(cfg.DIR_MODELS_CAFFE, 'face_detect', 'opencv_face_detector.prototxt')
    fp_model = join(cfg.DIR_MODELS_CAFFE, 'face_detect', 'opencv_face_detector.caffemodel')
    log.debug('fp_model: {}'.format(fp_model))
    net = cv.dnn.readNet(fp_prototxt, fp_model)
    # TODO parameterize
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
  elif  opt_net == types.FaceDetectNet.DLIB_CNN:
    # use dlib's CNN module
    cuda_visible_devices = os.getenv('CUDA_VISIBLE_DEVICES', '')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt_gpu)
    cnn_face_detector = dlib.cnn_face_detection_model_v1(cfg.DIR_MODELS_DLIB_CNN)
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices  # reset
  elif  opt_net == types.FaceDetectNet.DLIB_HOG:
    # use dlib's HoG module
    dlib_hog_predictor = dlib.get_frontal_face_detector()
  elif  opt_net == types.FaceDetectNet.HAAR:
    # use opencv's haarcasde module
    log.error('not yet implemented')
    return


  # ----------------------------------------------------------------
  # process

  # iterate sink
  while True:
    
    chair_item = yield
    
    metadata = {}
    
    for frame_idx, frame in chair_item.keyframes.items():

      rois = []

      if opt_net == types.FaceDetectNet.CVDNN:
        # use OpenCV's DNN face detector with caffe model
        frame = cv.resize(frame, opt_dnn_size)
        blob = cv.dnn.blobFromImage(frame, dnn_scale, opt_dnn_size, dnn_mean)
        net.setInput(blob)
        net_outputs = net.forward()

        for i in range(0, net_outputs.shape[2]):
          conf = net_outputs[0, 0, i, 2]
          if conf > opt_conf_thresh:
            rect_norm = net_outputs[0, 0, i, 3:7]
            rois.append( ROIDetectResult(conf, rect_norm) )
            log.debug('face roi: {}'.format(rect_norm))
              
        
      elif opt_net == types.FaceDetectNet.DLIB_CNN:
        frame = im_utils.resize(frame, width=opt_dnn_size[0], height=opt_dnn_size[1])
        # convert to RGB for dlib
        dim = frame.shape[:2][::-1]
        frame = im_utils.bgr2rgb(frame)
        mmod_rects = cnn_face_detector(frame, opt_pyramids)
        for mmod_rect in mmod_rects:
          if mmod_rect.confidence > opt_conf_thresh:
            bbox = BBox.from_dlib_dim(mmod_rect.rect, dim)
            # NB mmod_rect.confidence is sometimes > 1.0 ?
            rois.append( ROIDetectResult(mmod_rect.confidence, bbox.as_xyxy()) )

      elif opt_net == types.FaceDetectNet.DLIB_HOG:
        frame = im_utils.resize(frame, width=opt_dnn_size[0], height=opt_dnn_size[1])
        # convert to RGB for dlib
        dim = frame.shape[:2][::-1]
        frame = im_utils.bgr2rgb(frame)  # ?
        hog_results = dlib_hog_predictor.run(frame, opt_pyramids)
        if len(hog_results[0]) > 0:
          for rect, score, direction in zip(*hog_results):
            if score > opt_conf_thresh:
              bbox = BBox.from_dlib_dim(rect, dim)
              rois.append( ROIDetectResult(score, bbox.as_xyxy()) )

      
      
      metadata[frame_idx] = rois
        
    # append metadata to chair_item's mapping item
    chair_item.set_metadata(metadata_type, ROIMetadataItem(metadata))
  
    # ----------------------------------------------------------------
    # yield back to the processor pipeline

    # send back to generator
    sink.send(chair_item)


