"""Data representation for media item metadata attributes

Warning: if you change anything here you may need to regenerate the previous modeled data

"""
import os
from os.path import join
import json
import pickle
from pathlib import Path
from datetime import datetime
import logging

from dateutil import parser

from vframe.utils import file_utils
from vframe.settings import vframe_cfg as cfg
from vframe.settings.paths import Paths
from vframe.settings import types


class ClassifyResult:
  """Stores result from image classification processes"""
  def __init__(self, idx, score):
    self.idx = int(idx)
    self.score = float(score)

  def serialize(self):
    return {'idx': self.idx, 'score': self.score}
  

class DetectResult:
  """Stores result from image detection processes"""
  def __init__(self, idx, score, rect):
    self.rect = list(map(float, rect))  # normalized (x1, y1, x2, y2)
    self.idx = int(idx)
    self.score = float(score)

  def serialize(self):
    return {'idx': self.idx, 'score': self.score, 'rect': self.rect}


# ---------------------------------------------------------------------------
# Base class for MetadataItems
# ---------------------------------------------------------------------------
class MetadataItem(object):
  """Base class for MetadataItem sublcasses
  items[sha256]['metadata']['data_key'] = MetadataItem()
  """
  def __init__(self, metadata):
    self._metadata = metadata


  @classmethod
  def from_dict(cls, data):
    """Create Class object from dict data"""
    return cls(data)

  @classmethod
  def from_map(cls, data):
    """Create Class object from mapped data"""
    return cls(data)

  @classmethod
  def from_json(cls, data):
    """Create Class object from mapped JSON data"""
    # in this case same, but keep for flexibility
    return cls(data)

  @property
  def metadata(self):
    return self._metadata

  def serialize(self):
    """Returns serialized metadata"""
    return {k.name.lower(): v for k, v in self._metadata.items()}



class SugarcubeMetadataItem(MetadataItem):

  def __init__(self, fp, sa_id, md5):
    self._fp = fp
    self._sa_id = sa_id
    self._md5 = md5
    self._ext = file_utils.get_ext(fp)

  @classmethod
  def from_dict(cls, data):
    fp = data.get('fp', None)
    sa_id = data.get('sa_id', None)
    md5 = data.get('md5', None)
    return cls(fp, sa_id, md5)

  @property
  def filepath(self):
    return self._fp_media

  @property
  def sa_id(self):
    return self._sa_id
  
  @property
  def md5(self):
    return self._md5

  @property
  def ext(self):
    return self._ext
  

  def serialize(self):
    return {
      'fp': self._fp, 
      'sa_id': self._sa_id, 
      'md5': self._md5
      }
  
  
# ---------------------------------------------------------------------------
# Mediainfo Metadata: information about the media attributes
# ---------------------------------------------------------------------------
class MediainfoMetadataItem(MetadataItem):
  """Data representation for metadata extracted with mediainfo utility"""

  def __init__(self, video, audio=None):
    """
    """
    metadata = {
      types.MediainfoMetadata.VIDEO: video,
      types.MediainfoMetadata.AUDIO: audio
    }
    super().__init__(metadata)


  def parse_video_info(self, mediainfo):
    """Parses video info from JSONified mediainfo object
    
    These should all be Enum types eventually
    """

    v = {}
    v['color_primaries'] = str(mediainfo.get('color_primaries',''))  # meaning unsure
    v['frame_count'] = int(mediainfo.get('frame_count', 0))
    v['frame_rate'] = float(mediainfo.get('frame_rate', 0))
    v['bit_depth'] = int(mediainfo.get('bit_depth', 0))
    v['width'] = int(mediainfo.get('width', 0))
    v['height'] = int(mediainfo.get('height', 0))
    v['duration'] = float(mediainfo.get('duration', 0))
    v['aspect_ratio'] = float(mediainfo.get('display_aspect_ratio', 0))
    v['internet_media_type'] = str(mediainfo.get('internet_media_type', ''))  # "mediainfo/h264"
    v['bit_rate'] = int(mediainfo.get('bit_rate', 0))  # 802956 bytes/sec
    v['codec_id'] = str(mediainfo.get('codec_id', ''))  # avc1
    v['frame_rate_mode'] = str(mediainfo.get('frame_rate_mode', ''))  # CFR
    v['stream_size'] = int(mediainfo.get('stream_size', 0))  # 8294537 (bytes'])
    # convert encoded date timestamp to ISO-8601
    encoded_date = mediainfo.get('encoded_date','')  # "UTC 2018-01-29 20:40:57"
    if encoded_date:
      try:
        tz = encoded_date[:3].strip()
        dt = parser.parse(encoded_date[3:].strip())
        v['encoded_date'] = dt.isoformat()
      except:
        v['encoded_date'] = ''
    else:
      v['encoded_date'] = ''

    # convert tagged date timestamp to ISO-8601
    tagged_date = mediainfo.get('tagged_date', '')
    if tagged_date:
      try:
        dt = parser.parse(tagged_date[3:].strip())
        v['tagged_date'] = dt.isoformat()  # "UTC 2018-01-29 20:40:57"
      except:
        v['tagged_date'] = ''
    else:
      v['tagged_date'] = ''

    return v

  def parse_audio_info(self, mediainfo):
    """Parses audio info from JSONified mediainfo object"""
    a = {}
    a['frame_rate'] = float(mediainfo.get('frame_rate', 0))
    a['samples_count'] = int(mediainfo.get('samples_count', 0))
    a['codec'] = str(mediainfo.get('codec',''))
    a['codec_id'] = str(mediainfo.get('codec_id', ''))
    a['codec_cc'] = str(mediainfo.get('codec_cc', ''))
    a['duration'] = float(mediainfo.get('duration', 0))  # milliseconds?
    a['frame_count'] = int(mediainfo.get('frame_count', 0))  # unsure meaning
    a['codec_family'] = str(mediainfo.get('codec_family', ''))  # AAC
    # mediainfo audio can have multiple channels
    try:
      a['channels'] = {
        'sampling_rates': list(map(int, str(mediainfo.get('sampling_rate', '')).split('/'))),
        'channel_s': list(map(int, str(mediainfo.get('channel_s', '')).split('/')))
        }
    except:
      a['channels'] = {}
      # logger.info('sampling_rate: {}'.format(mediainfo.get('sampling_rate', '')))
      # logger.info('channel_s: {}'.format(mediainfo.get('channel_s', '')))
    a['samples_per_frame'] = int(mediainfo.get('samples_per_frame', 0))  # 1024
    # convert to ISO-8601
    encoded_date = mediainfo.get('encoded_date', '')
    try:
      dt = parser.parse(encoded_date[3:].strip())
      a['encoded_date'] = dt.isoformat()
    except:
      a['encoded_date'] = ''
    return a

  @classmethod
  def from_index_json(cls, json_data):
    """Returns MediainfoMetadataItem using raw output from pymediainfo
    These should all be Enum types eventually
    """
    log = logging.getLogger('vframe')
    media_info_video = json_data.get('video', {})
    media_info_audio = json_data.get('audio', {})
    try:
      audio = cls.parse_audio_info(cls, media_info_audio)
    except:
      audio = {}
      log.error('error parsing audio')
      log.error('audio: {}'.format(media_info_audio))
    try:
      video = cls.parse_video_info(cls, media_info_video)
    except:
      log.error('error parsing video')
      log.error('video: {}'.format(media_info_video))
      video = {}
    return cls(video, audio=audio)

  @classmethod
  def from_dict(cls, data):
    return cls(data.get('video', {}), audio=data.get('audio', {}))

  @classmethod
  def from_map(cls, data):
    video = data.get(types.MediainfoMetadata.VIDEO, {})
    audio = data.get(types.MediainfoMetadata.AUDIO, {})
    return cls(video, audio)



# ---------------------------------------------------------------------------
# Keyframe metadata: stores indices of representative keyframes
# ---------------------------------------------------------------------------
class KeyframeStatusMetadataItem(MetadataItem):
  """Slim data representation for metadata extracted with mediainfo utility
  Unlike other metadata items, this object gathers data from others
  and is used for file filtering. Functions a helper/bridge object

  Currently: Gathers data from MEDIAINFO, and keyframe file existence
  """

  def __init__(self, metadata):
    super().__init__(metadata)

  def get_status(self, opt_size):
    return self._metadata.get(opt_size, {})

  @classmethod
  def from_dict(cls, data):
    """From serialized dict"""
    metadata = {
      types.ImageSize.THUMB: bool(data[types.ImageSize.THUMB.name.lower()]),
      types.ImageSize.SMALL: bool(data[types.ImageSize.SMALL.name.lower()]),
      types.ImageSize.MEDIUM: bool(data[types.ImageSize.MEDIUM.name.lower()]),
      types.ImageSize.LARGE: bool(data[types.ImageSize.LARGE.name.lower()]),
    }
    return cls(metadata)
  


# ---------------------------------------------------------------------------
# Keyframe metadata: stores indices of representative keyframes
# ---------------------------------------------------------------------------
class KeyframeMetadataItem(MetadataItem):

  def __init__(self, dense, basic, expanded, generated=False):
    metadata = {
      types.KeyframeMetadata.DENSE: dense,
      types.KeyframeMetadata.BASIC: basic,
      types.KeyframeMetadata.EXPANDED: expanded
      }
    super().__init__(metadata)

  def get_keyframes(self, density):
    return self._metadata[density]

  @classmethod
  def from_index_json(cls, json_data):
    """Returns a self.Class object using the index.json generated files"""
    # it's possible that the JSON object is an empty {}
    dense = json_data.get('dense', {})
    basic = json_data.get('basic', {})
    expanded = json_data.get('expanded', {})
    return cls(dense, basic, expanded)

  @classmethod
  def from_dict(cls, data):
    """Returns self.Class object using the metadata in the serialized summary JSON files"""
    dense = data.get('dense', {})
    basic = data.get('basic', {})
    expanded = data.get('expanded', {})
    generated = data.get('generated')
    return cls(dense, basic, expanded)

  # TODO improve from_* methods
  @classmethod
  def from_map(cls, data):
    """Returns self.Class object using the metadata in the serialized summary JSON files"""
    dense = data.get(types.KeyframeMetadata.DENSE, {})
    basic = data.get(types.KeyframeMetadata.BASIC, {})
    expanded = data.get(types.KeyframeMetadata.EXPANDED, {})
    return cls(dense, basic, expanded)



# ---------------------------------------------------------------------------
# Feature vector metadata
# ---------------------------------------------------------------------------
class FeatureMetadataItem(MetadataItem):
  """Stores all the feature vectors"""
  def __init__(self, metadata):
    super().__init__(metadata)

  def serialize(self):
    """Returns serialized metadata"""
    return self._metadata


# ---------------------------------------------------------------------------
# Keyframe metadata: stores indices of representative keyframes
# ---------------------------------------------------------------------------
class ClassifyMetadataItem(MetadataItem):

  def __init__(self, metadata):
    """Represents classification results from Places365 DNN
    :param metadata: (list) of (ClassifyResult)"""
    super().__init__(metadata)

  @classmethod
  def from_json(cls, data):
    # metadata = {k: ClassifyResult(v['idx'], v['score']) for k, v in data.items()}
    return cls(data)

  @classmethod
  def from_dict(cls, data):
    metadata = {}
    for frame_idx, classifications in data.items():
      metadata[frame_idx] = [ClassifyResult(x['idx'], x['score']) for x in classifications]
    return cls(metadata)

  @classmethod
  def from_map(cls, data):
    return cls(data)
    
  def serialize(self):
    """Returns serialized metadata"""
    metadata = {} 
    for k, v in self._metadata.items():
      metadata[k] = [x.serialize() for x in v]
    return metadata


class DetectMetadataItem(MetadataItem):

  def __init__(self, metadata):
    """Represents classification results from Places365 DNN
    :param metadata: (list) of (DetectResult)"""
    super().__init__(metadata)

  @classmethod
  def from_json(cls, data):
    # metadata = {k: DetectResult(v['idx'], v['score']) for k, v in data.items()}
    return cls(data)

  @classmethod
  def from_dict(cls, data):
    metadata = {}
    for frame_idx, detections in data.items():
      metadata[frame_idx] = [DetectResult(x['idx'], x['score'], x['rect']) for x in detections]
    return cls(metadata)
    
  def serialize(self):
    """Returns serialized metadata"""
    metadata = {} 
    for k, v in self._metadata.items():
      metadata[k] = [x.serialize() for x in v]
    return metadata
