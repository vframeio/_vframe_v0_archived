class BBox:

  def __init__(self, x1, y1, x2, y2, imh, imw):
    """Initialize BBox vars"""
    self._x1 = x1
    self._y1 = y1
    self._x2 = x2
    self._y2 = y2
    self._width = x2 - x1
    self._height = y2 - y1
    self._image_width = imw
    self._image_height = imh
    self._cx = x1 - self._width / 2
    self._cy = y1 - self._height / 2
    # normalize
    self._cx_norm = self._cx / imw
    self._cy_norm = self._cy / imh
    self._width_norm = self._width / imw
    self._height_norm = self._height / imh

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
  def image_width(self):
    return self._image_width
  
  @property
  def image_height(self):
    return self._image_height
  
  
  # -----------------------------------------------------------------
  # Convert to

  def as_haar(self):
    """Converts BBox back to haar type"""
    return (self._x1, self._y1, self._x2 - self._x1, self._y2 - self._y1)

  def as_css(self):
    """Converts BBox to CSS (top, right, bottom, left)""" 
    return (self._y1, self._x2, self._y2, self._x1)

  def as_dlib_rect(self):
    """Converts BBox to dlib rect type"""
    # TODO
    pass

  def as_yolo(self):
    """Converts BBox to normalized center x, center y, w, h"""
    return (self._cx_norm, self._cy_norm, self._width_norm, self._height_norm)

  def as_normalized(self):
    """Returns normalized x1, y1, x2, y2"""
    return (self._x1_norm, self._y1_norm, self._x2_norm, self._y2_norm)


  # -----------------------------------------------------------------
  # Create from

  @classmethod
  def from_haar(cls, rect):
    """OpenCV returns x1, y1, w, h"""
    return cls(rect[0], rect[1], rect[0] + rect[2], rect[3] + rect[1])

  @classmethod
  def from_css(cls, rect):
    """Converts rect from CSS (top, right, bottom, left)""" 
    return cls(rect[3], rect[0], rect[1], rect[2])

  @classmethod
  def from_dlib_rect(cls, box):
    """TODO"""
    pass
