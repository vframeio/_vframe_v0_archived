"""
Drawing utlities
"""

from vframe.utils import im_utils, file_utils, logger_utils
import cv2 as cv
import numpy as np
from PIL import Image, ImageOps, ImageFilter


log = logger_utils.Logger.getLogger()


def draw_rectangle_pil(draw_ctx, bbox, color=(0,255,0), width=1):
  for i in range(width):
    rect_start = (bbox.x1 - i, bbox.y1 - i)
    rect_end = (bbox.x2 + i, bbox.y2 + i)
    draw_ctx.rectangle((rect_start, rect_end), outline = color)