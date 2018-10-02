class YoloAnnoItem:

  def __init__(self, fn, class_id, r):
    """YOLO formatted annotation"""
    # each annotation is described as
    # class_index center_x center_y width height
    # 11 0.34419263456090654 0.611 0.4164305949008499 0.262
    
    self._filename = fn
    self.class_id = class_id
    nx, ny, nw, nh = list(map(float, (r['x'], r['y'], r['width'], r['height'])))
    self.nx = nx
    self.ny = ny
    self.nw = nw
    self.nh = nh
    ncx, ncy = (nx + nw / 2, ny + nh / 2)
    self.ncx = ncx
    self.ncy = ncy

  def as_line(self):
    """Returns annotation formatted for YOLO train.txt files"""
    # class_id center_x center_y width height
    # normalized values
    return ' '.join(list(map(str, [self.class_id, self.ncx, self.ncy, self.nw, self.nh] )))

  @property
  def filename(self):
    return self._filename

