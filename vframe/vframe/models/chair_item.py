"""Object that passes through the chair/click pipeline"""

from threading import Thread

import cv2 as cv
from imutils.video import FileVideoStream

from vframe.settings import types
from vframe.settings import vframe_cfg as cfg
from vframe.models.media_item import MediaRecordItem
from vframe.utils import logger_utils

import numpy as np

class ChairItem(object):

  _chair_type = None

  def __init__(self, ctx):
    """Items that pass through the chair pipeline"""
    self._ctx = ctx
    self._keyframes = {}
    self._drawframes = {}
    self.log = logger_utils.Logger.getLogger()


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
  def chair_type(self):
    return self._chair_type
  
  @property
  def media_format(self):
    """alternate name for same data"""
    return self._media_record.media_format
  

class VideoKeyframeChairItem(ChairItem):

  chair_type = types.ChairItemType.VIDEO_KEYFRAME  

  def __init__(self, ctx, frame, frame_idx):
    super().__init__(ctx)
    self._frame = frame
    self._frame_idx = frame_idx
    self._drawframe = frame.copy()
    self._keyframes = {frame_idx: self._frame}
    self._drawframes = {frame_idx: self._drawframe}
    mr = MediaRecordItem(0, types.MediaFormat.VIDEO, 0, metadata={})
    self._media_record = mr

  def remove_frames(self):
    self.remove_frame()
    self.remove_drawframe()

  def remove_frame(self):
    self._frame = None

  def remove_drawframe(self):
    self._drawframe = None

  @property
  def frame(self):
    return self._frame
  
  @property
  def drawframe(self):
    return self._drawframe
  


class PhotoChairItem(ChairItem):

  chair_type = types.ChairItemType.PHOTO

  def __init__(self, ctx, frame):

    self._ctx = ctx
    self._frame = frame
    self._drawframe = frame.copy()
    mr = MediaRecordItem(0, types.MediaFormat.VIDEO, 0, metadata={})
    self._media_record = mr


class VideoChairItem(ChairItem):

  chair_type = types.ChairItemType.VIDEO

  def __init__(self, ctx, fp_video):
    super().__init__(ctx)
    self._fp_video = fp_video
    self._stream = None
    self._frame_count = 0
    self._width = 0
    self._height = 0
    self._fps = 0
    self._mspf = 0
    self._last_display_ms = 0
    mr = MediaRecordItem(0, types.MediaFormat.VIDEO, 0, metadata={})
    self._media_record = mr


  def load_video_keyframes(self, opt_drawframes=False):
    """Loads keyframes from video"""
    self.log.debug('init load_video_keyframes')
    self._opt_drawframes = opt_drawframes
    
    self.log.debug('load: {}'.format(self._fp_video))
    # self._filevideostream = FileVideoStream(self._fp_video, transform=None, queueSize=256)

    # self._filevideostream.start()
    self.log.debug('filevideostream started')
    self._stream = cv.VideoCapture(self._fp_video)
    # _stream = self._filevideostream.stream
    self._frame_count = int(self._stream.get(cv.CAP_PROP_FRAME_COUNT))
    self._width = int(self._stream.get(cv.CAP_PROP_FRAME_WIDTH))
    self._height = int(self._stream.get(cv.CAP_PROP_FRAME_HEIGHT))
    self._fps = self._stream.get(cv.CAP_PROP_FPS)
    self._mspf = int(1 / self._fps * 1000)  # milliseconds per frame
    self.log.debug('frame_count: {}'.format(self._frame_count))
    self._stream.release()
    im_blank = np.zeros([720, 1280, 3],dtype=np.uint8)
    for i in range(self._frame_count):
      self._keyframes[i] = im_blank.copy()
      self._drawframes[i] = im_blank.copy()

    self.log.debug('start load thread')
    # make threaded
    self.load_thread = Thread(target=self.update_thread, args=())
    self.load_thread.daemon = True
    self.log.debug('really start load thread')
    try:
      self.load_thread.start()
    except Exception as ex:
      self.error('{}'.format(ex))


  def update_thread(self):
    
    self._stream = cv.VideoCapture(self._fp_video)
    valid, frame = self._stream.read()
    self.log.debug('size: {}'.format(frame.shape))

    self.log.debug('init update_thread')
    frame_idx = 0
    
    while True:
      valid, frame = self._stream.read()
      if not valid:
        self._stream.release()
        break
      self._keyframes[frame_idx] = frame
      if self._opt_drawframes:
        self._drawframes[frame_idx] = frame.copy()  # make drawable copy
      frame_idx += 1

  @property
  def last_display_ms(self):
    return self._last_display_ms
  
  @last_display_ms.setter
  def last_display_ms(self, value):
    self._last_display_ms = value

  @property
  def mspf(self):
    return self._mspf
  
  @property
  def width(self):
    return self._width
  
  @property
  def height(self):
    return self._height
  
  @property
  def fps(self):
    return self._fps
   
  @property
  def frame_count(self):
    return self._frame_count
   
  @property 
  def filevideostream(self):
    return self._filevideostream
  

  @property
  def drawframe(self):
    return self._drawframe



class MediaRecordChairItem(ChairItem):

  chair_type = types.ChairItemType.MEDIA_RECORD

  def __init__(self, ctx, media_record):
    super().__init__(ctx)
    self._media_record = media_record
    self._sha256 = self._media_record.sha256

  @property
  def sha256(self):
    return self._sha256

  @property
  def verified(self):
    return self._media_record.verified

  @property
  def media_record(self):
    """both refer to same data"""
    return self._media_record
