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
ty_offset = 6
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
  pt_tl, pt_br = bbox.pt_tl, bbox.pt_br
  cv.rectangle(frame, pt_tl.tuple() , pt_br.tuple(), rect_color, thickness=stroke_weight)

  # prepare label
  if text:
    label = '{} ({:.2f})'.format(text, float(score))
  else:
    label = '{:.2f}'.format(float(score))

  tw, th = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, tx_scale, tx_weight)[0]

  # draw label bg
  bg_y = pt_tl.y - th - ty2_offset  # text above box default
  txt_y = 0 -  int(1.5 * ty_offset)
  if bg_y < 0:
    bg_y = pt_tl.y + th + ty2_offset  # text below box default
    txt_y = int(2.4 * ty_offset)

  rect_pt_br = (pt_tl.x + tw + tx2_offset, bg_y)
  cv.rectangle(frame, pt_tl.tuple(), rect_pt_br, rect_color, -1)
  
  # draw label
  cv.putText(frame, label, pt_tl.offset(tx_offset, -ty_offset), font, tx_scale, text_color, tx_weight)
  return frame


def draw_detection_result(frame, classes, detection_result, imw, imh, 
  stroke_weight=2, rect_color=(0,255,0), text_color=(0,0,0)):
  
  bbox = BBox(*detection_result.rect).to_dim((imw, imh))
  class_idx = detection_result.idx
  score = detection_result.score

  lum = (0.2126 * rect_color[2]) + (0.7152 * rect_color[1]) + (0.0722 * rect_color[0])
  if lum > 100:
    text_color = (0,0,0)
  else:
    text_color = (255, 255, 255)
  # draw border
  pt_tl, pt_br = bbox.pt_tl, bbox.pt_br
  cv.rectangle(frame, pt_tl.tuple() , pt_br.tuple(), rect_color, thickness=stroke_weight)

  # prepare label
  class_label = classes[class_idx].upper().replace('_',' ')
  label = '{} ({:.2f})'.format(class_label, float(score))
  tw, th = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, tx_scale, tx_weight)[0]

  # draw label bg
  bg_y = pt_tl.y - th - ty2_offset  # text above box default
  txt_y = 0 -  int(1.5 * ty_offset)
  if bg_y < 0:
    bg_y = pt_tl.y + th + ty2_offset  # text below box default
    txt_y = int(2.4 * ty_offset)

  rect_pt_br = (pt_tl.x + tw + tx2_offset, bg_y)
  cv.rectangle(frame, pt_tl.tuple(), rect_pt_br, rect_color, -1)

  # draw label
  cv.putText(frame, label, pt_tl.offset(tx_offset, txt_y), font, tx_scale, text_color, tx_weight)
  return frame