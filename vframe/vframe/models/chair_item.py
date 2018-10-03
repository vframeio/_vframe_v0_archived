"""Object that passes through the chair/click pipeline"""

class ChairItem(object):

  def __init__(self, ctx, media_record):
    """Items that pass through the chair pipeline"""
    self._ctx = ctx
    self._media_record = media_record
    self._sha256 = self._media_record.sha256
    self._keyframes = {}
    self._drawframes = {}


  def set_keyframes(self, keyframes, add_drawframe=False):
    """Adds dict of keyframe images"""
    self._keyframes = keyframes
    if add_drawframe:
      self._drawframes = keyframes.copy()

  def remove_keyframes(self):
    self._keyframes = {}

  def set_drawframes(self, keyframes):
    """Adds dict of keyframe images"""
    self._keyframes = keyframes

  def remove_drawframes(self):
    self._drawframes = {}
    

  # shortcuts
  def get_metadata(self, metadata_type):
    """Gets metadata dict if it exists. Returns empty dict if none"""
    return self._media_record.get_metadata(metadata_type)

  def set_metadata(self, metadata_type, metadata):
    self._media_record.set_metadata(metadata_type, metadata)

  @property
  def keyframes(self):
    return self._keyframes

  @property
  def keyframe(self, frame_idx):
    """Returns keyframe image from frame index if exists"""
    return self._keyframes.get(frame_idx, None)

  @property
  def drawframes(self):
    return self._drawframes

  @property
  def drawframe(self, frame_idx):
    """Returns keyframe image from frame index if exists"""
    return self._drawframes.get(frame_idx, None)
  
  @property
  def ctx(self):
    return self._ctx

  @property
  def item(self):
    return self._media_record

  @property
  def record(self):
    """both refer to same data"""
    return self._media_record

  @property
  def media_record(self):
    """both refer to same data"""
    return self._media_record

  @property
  def sha256(self):
    return self._sha256

  @property
  def verified(self):
    return self._media_record.verified

  @property
  def format(self):
    return self._media_record.media_format

  @property
  def media_format(self):
    """alternate name for same data"""
    return self._media_record.media_format
  