"""Object to represent media records
"""
import logging

from vframe.utils import click_utils
from vframe.settings import vframe_cfg as cfg
from vframe.settings import types

from vframe.models.metadata_item import KeyframeMetadataItem, MediainfoMetadataItem
from vframe.models.metadata_item import KeyframeStatusMetadataItem, ClassifyMetadataItem
from vframe.models.metadata_item import FeatureMetadataItem, DetectMetadataItem
from vframe.models.metadata_item import SugarcubeMetadataItem

class MediaRecordItem:
  """Represents a media record (video or photo) in the VFRAME system"""
  
  log = logging.getLogger(cfg.LOGGER_NAME)

  def __init__(self, sha256, media_format, verified, metadata={}):
    self._sha256 = sha256
    self._media_format = media_format  # types.MediaFormat
    self._verified = verified  # types.Verified
    self._metadata = metadata

  # --------------------------------------------------------------
  # metadata
  # --------------------------------------------------------------
  def set_metadata(self, metadata_type, metadata):
    self._metadata[metadata_type] = metadata

  def remove_metadata(self, metadata_type):
    """Removes a metadata element"""
    _ = self._metadata.pop(metadata_type, None)

  def get_metadata(self, metadata_type):
    """Returns metadata values if exist
    :param metadata_type: (Enum.MetadataType) 
    """
    if metadata_type in self._metadata.keys():
      return self._metadata.get(metadata_type)
    else:
      return {}
  

  # --------------------------------------------------------------
  # properties
  # --------------------------------------------------------------
  @property
  def metadata(self):
    return self._metadata

  @property
  def sha256(self):
    return self._sha256
  
  @property
  def media_format(self):
    """MediaFormat.VIDEO or MediaFormat.PHOTO"""
    return self._media_format

  @property
  def verified(self):
    return self._verified
  
  @classmethod
  def map_metadata(cls, metadata):
    """Maps metadata into object classes using enum dict keys"""
    # k = MediainfoMetadataType, KeyframeMetadataType
    # TODO remap this into dict of objects
    mapped = {}
    for k, v in metadata.items():
      if k == types.Metadata.SUGARCUBE.name.lower():
        mapped[types.Metadata.SUGARCUBE] = SugarcubeMetadataItem.from_dict(v)
      elif k == types.Metadata.MEDIAINFO.name.lower():
        mapped[types.Metadata.MEDIAINFO] = MediainfoMetadataItem.from_dict(v)
      elif k == types.Metadata.KEYFRAME.name.lower():
        mapped[types.Metadata.KEYFRAME] = KeyframeMetadataItem.from_dict(v) 
      elif k == types.Metadata.KEYFRAME_STATUS.name.lower():
        mapped[types.Metadata.KEYFRAME_STATUS] = KeyframeStatusMetadataItem.from_dict(v) 
      elif k == types.Metadata.FEATURE_VGG16.name.lower():
        mapped[types.Metadata.FEATURE_VGG16] = FeatureMetadataItem.from_dict(v) 
      elif k == types.Metadata.PLACES365.name.lower():
        mapped[types.Metadata.PLACES365] = ClassifyMetadataItem.from_dict(v) 
      elif k == types.Metadata.COCO.name.lower():
        mapped[types.Metadata.COCO] = DetectMetadataItem.from_dict(v) 
      else:
        msg = '{} is a not valid metadata type or not yet impelemented'.format(k) 
        cls.log.error(msg)
        raise ValueError(msg)
    return mapped

    
  @classmethod
  def from_dict(cls, data):
    """Convert serialized data from JSON/Pickle into mapped enum data"""
    # TODO might be slow, use precomputed lookup table instead
    media_format = types.find_type(data['media_format'], types.MediaFormat)
    verified = types.find_type(data['verified'], types.Verified)
    metadata = cls.map_metadata(data.get('metadata', {}))
    return cls(data['sha256'], media_format, verified, metadata=metadata)



  # --------------------------------------------------------------
  # seralize to JSON/Pickle
  # --------------------------------------------------------------
  def serialize(self):
    """JSON representation"""
    return {
      'metadata': {k.name.lower(): v.serialize() for \
        k, v in self._metadata.items() if v},
      'sha256': self._sha256, 
      'media_format': self._media_format.name.lower(),
      'verified': self._verified.name.lower()
    }


