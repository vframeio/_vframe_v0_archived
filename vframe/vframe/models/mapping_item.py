from pathlib import Path
import logging

from vframe.settings import vframe_cfg as cfg
from vframe.utils import file_utils
from vframe.models.metadata_item import KeyframeMetadataItem, MediainfoMetadataItem
from vframe.models.metadata_item import KeyframeStatusMetadataItem, ClassifyMetadataItem
from vframe.models.metadata_item import FeatureMetadataItem, DetectMetadataItem
from vframe.settings import types


class MediaRecord:
  """Represents a media record (video or photo) in the VFRAME system"""
  def __init__(self, sha256, opt_media_type):
    self._sha256 = sha256
    self._media_type = opt_media_type

  @property
  def sha256(self):
    return self._sha256
  
  @property
  def media_type(self):
    """MediaType.VIDEO or MediaType.PHOTO"""
    return self._media_type
    


class MappingItem:
  """Represents mapping between generic filesytem object and VFRAME object

  Mapping items are evidential and never change.
  They require only a SHA256 and file type extension
  Because local system filepaths change often, the MediaItem is added later
  """
  
  _root_media_item = None
  _media = {}  # SHA256 key
  _metadata = {}  # SHA256 key

  def __init__(self, sha256, ext):
    self._sha256 = sha256
    self._ext = ext
    self._media_type = file_utils.media_type_ext(ext) # enums.MediaType

  def set_main_media(self, dir_src):
    """Sets a main MediaItem for the mapping object"""
    self.main_media_item = MediaItem(dir_src, sha256, ext)

  def add_metadata(self, metadata_type, metadata):
    """Sets mapping items metadata property for metadata_type
    If current metadata exists, is overwritten
    """
    self._metadata[metadata_type] = metadata

  def remove_metadata(self, metadata_type):
    """Removes a metadata element"""
    _ = self._metadata.pop(metadata_type, None)

  def route_source(self, dir_media_src):
    """Appends the source of the media items parent directory
    """
    self._dir_media_src = dir_src

  def get_metadata(self, metadata_type):
    """Returns metadata values if exist
    :param metadata_type: (Enum.MetadataType) 
    """
    # TODO change to get_metadata_type
    try:
      metadata_item = self._metadata.get(metadata_type,{})
      metadata = metadata_item.metadata
    except:
      metadata = {}
    return metadata
    
  @property
  def sha256(self):
    return self._sha256
  
  @property
  def ext(self):
    return self._ext

  @property
  def media_type(self):
    return self._media_type

  @property
  def metadata(self):
    return self._metadata
  
  @property
  def media(self):
    return self._media
  
  def serialize(self):
    return {
      'sha256': self._sha256, 
      'ext': self._ext, 
      'metadata': {k.name.lower(): v.serialize() for k, v in self._metadata.items() if v}
      }
  


