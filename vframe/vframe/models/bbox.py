from dlib import rectangle as dlib_rectangle
import numpy as np

class BBoxPoint:

  def __init__(self, x, y):
    self._x = x
    self._y = y

  @property
  def x(self):
    return self._x
  
  @property
  def y(self):
    return self._y
  
  def offset(self, x, y):
    return (self._x + x, self._y + y)

  def tuple(self):
    return (self._x, self._y)


class BBox:

  def __init__(self, x1, y1, x2, y2):
    """Represents a bounding box and provides methods for accessing and modifying
    :param x1: normalized left coord
    :param y1: normalized top coord
    :param x2: normalized right coord
    :param y2: normalized bottom coord
    """
    self._x1 = x1
    self._y1 = y1
    self._x2 = x2
    self._y2 = y2
    self._width = x2 - x1
    self._height = y2 - y1
    self._cx = x2 - (self._width / 2)
    self._cy = y2 - (self._height / 2)
    self._bbp1 = BBoxPoint(x1, y1)
    self._bbp2 = BBoxPoint(x2, y2)
    self._rect = (self._x1, self._y1, self._x2, self._y2)


  @property
  def pt_tl(self):
    return self._bbp1

  @property
  def pt_br(self):
    return self._bbp2

  @property
  def x1(self):
    return self._x1
  
  @property
  def y1(self):
    return self._y1

  @property
  def x2(self):
    return self._x2
  
  @property
  def y2(self):
    return self._y2
  
  @property
  def height(self):
    return self._height

  @property
  def width(self):
    return self._width

  @property
  def h(self):
    return self._height

  @property
  def w(self):
    return self._width
    

  # -----------------------------------------------------------------
  # Convert to

  def to_dim(self, dim):
    """scale is (w, h) is tuple of dimensions"""
    w, h = dim
    rect = list((np.array(self._rect) * np.array([w, h, w, h])).astype('int'))
    return BBox(*rect)

  def normalize(self, rect, dim):
    w, h = dim
    x1, y1, x2, y2 = rect
    return (x1 / w, y1 / h, x2 / w, y2 / h)

  # -----------------------------------------------------------------
  # Format as

  def as_xyxy(self):
    """Converts BBox back to x1, y1, x2, y2 rect"""
    return (self._x1, self._y1, self._x2, self._y2)

  def as_xywh(self):
    """Converts BBox back to haar type"""
    return (self._x1, self._y1, self._width, self._height)

  def as_trbl(self):
    """Converts BBox to CSS (top, right, bottom, left)""" 
    return (self._y1, self._x2, self._y2, self._x1)

  def as_dlib(self):
    """Converts BBox to dlib rect type"""
    return dlib.rectangle(self._x1, self._y1, self._x2, self._y2)

  def as_yolo(self):
    """Converts BBox to normalized center x, center y, w, h"""
    return (self._cx, self._cy, self._width, self._height)


  # -----------------------------------------------------------------
  # Create from

  @classmethod
  def from_xyxy_dim(cls, x1, y1, x2, y2, dim):
    """Converts x1, y1, w, h to BBox and normalizes
    :returns BBox
    """
    rect = cls.normalize(cls, (x1, y1, x2, y2), dim)
    return cls(*rect)

  @classmethod
  def from_xywh_dim(cls, x, y, w, h, dim):
    """Converts x1, y1, w, h to BBox and normalizes
    :param rect: (list) x1, y1, w, h
    :param dim: (list) w, h
    :returns BBox
    """
    rect = cls.normalize(cls, (x, y, x + w, y + h), dim)
    return cls(*rect)

  @classmethod
  def from_xywh(cls, rect):
    """Converts x1, y1, w, h to BBox
    :param rect: (list) x1, y1, w, h
    :param dim: (list) w, h
    :returns BBox
    """
    rect = (rect[0], rect[1], rect[0] + rect[2], rect[3] + rect[1])
    return cls(rect)

  @classmethod
  def from_css(cls, rect, dim):
    """Converts rect from CSS (top, right, bottom, left) to BBox
    :param rect: (list) x1, y1, x2, y2
    :param dim: (list) w, h
    :returns BBox
    """
    rect = (rect[3], rect[0], rect[1], rect[2])
    rect = cls.normalize(cls, rect, dim)
    return cls(*rect)

  @classmethod
  def from_dlib_dim(cls, rect, dim):
    """Converts dlib.rectangle to BBox
    :param rect: (list) x1, y1, x2, y2
    :param dim: (list) w, h
    :returns dlib.rectangle
    """ 
    rect = (rect.left(), rect.top(), rect.right(), rect.bottom())
    rect = cls.normalize(cls, rect, dim)
    return cls(*rect)


  def str(self):
    """Return BBox as a string "x1, y1, x2, y2" """
    return self.as_box()

