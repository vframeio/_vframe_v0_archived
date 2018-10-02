"""Object to represent media input/output items
"""

class VideoQuality:
  # Defines minimum threshold for a video quality score
  def __init__(self, w, h, frame_rate, duration, codec='avc1'):
    self.width = w
    self.height = h
    self.frame_rate = frame_rate
    self.duration = duration  # seconds
    self.frame_count = duration * frame_rate
    self.codec = codec