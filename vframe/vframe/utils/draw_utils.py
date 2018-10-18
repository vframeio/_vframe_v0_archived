"""
Drawing utlities
"""
import cv2 as cv
import numpy as np
from PIL import Image, ImageOps, ImageFilter

from vframe.models.bbox import BBox
from vframe.utils import im_utils, file_utils, logger_utils


log = logger_utils.Logger.getLogger()


font = cv.FONT_HERSHEY_SIMPLEX
tx_offset = 4
ty_offset = 5
tx2_offset = 2 * tx_offset
ty2_offset = 2 * ty_offset
tx_scale = 0.4
tx_clr = (0,0,0)
tx_weight = 1


def draw_rectangle_pil(draw_ctx, bbox, color=(0,255,0), width=1):
  for i in range(width):
    rect_start = (bbox.x1 - i, bbox.y1 - i)
    rect_end = (bbox.x2 + i, bbox.y2 + i)
    draw_ctx.rectangle((rect_start, rect_end), outline = color)

def draw_roi(frame, detection_result, imw, imh, text=None,
  stroke_weight=2, rect_color=(0,255,0),text_color=(0,0,0)):
  
  dim = (imw, imh)
  bbox = BBox(*detection_result.rect).to_dim(dim)
  score = detection_result.score

  # draw border
  pt1, pt2 = bbox.pt1, bbox.pt2
  cv.rectangle(frame, pt1.tuple() , pt2.tuple(), rect_color, thickness=stroke_weight)

  # prepare label
  if text:
    label = '{} ({:.2f})'.format(text, float(score))
  else:
    label = '{:.2f}'.format(float(score))

  tw, th = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, tx_scale, tx_weight)[0]

  # draw label bg
  rect_pt2 = (pt1.x + tw + tx2_offset, pt1.y - th - ty2_offset)
  cv.rectangle(frame, pt1.tuple(), rect_pt2, rect_color, -1)
  # draw label
  cv.putText(frame, label, pt1.offset(tx_offset, -ty_offset), font, tx_scale, tx_clr, tx_weight)
  return frame


def draw_detection_result(frame, classes, detection_result, imw, imh, 
  stroke_weight=2, rect_color=(0,255,0),text_color=(0,0,0)):
  
  bbox = BBox(detection_result.rect).scale(imw, imh)
  class_idx = detection_result.idx
  score = detection_result.score

  # draw border
  pt1, pt2 = bbox.pt1, bbox.pt2
  cv.rectangle(frame, pt1.tuple() , pt2.tuple(), rect_color, thickness=stroke_weight)

  # prepare label
  label = '{} ({:.2f})'.format(classes[class_idx].upper(), float(score))
  log.debug('label: {}, bbox: {}'.format(label, str(bbox.as_box())))
  tw, th = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, tx_scale, tx_weight)[0]

  # draw label bg
  rect_pt2 = (pt1.x + tw + tx2_offset, pt1.y + th + ty2_offset)
  cv.rectangle(frame, pt1.tuple(), rect_pt2, rect_color, -1)
  # draw label
  cv.putText(frame, label, pt1.offset(tx_offset, 3*ty_offset), font, tx_scale, tx_clr, tx_weight)
  return frame