from enum import Enum

def find_type(name, enum_type):
  for enum_opt in enum_type:
    if name == enum_opt.name.lower():
      return enum_opt
  return None


# ---------------------------------------------------------------------
# Metadata types
# ---------------------------------------------------------------------

class Metadata(Enum):
  """All types of metadata require unique enum type
  MEDIAINFO: data extracted from pymediainfo (edited, but verbose)
  EXIF: data extracted from pyexif (relates to camera type)
  KEYFRAME: the representative keyframes
  KEYFRAME_STATUS: file status on system and basic attributes
  KEYFRAME: keyframe positions extracted from a video
  
  Not yet implemented:
  PLACES365: standard 365 place attribute metadata
  YOLO9000: standard Yolo9000 object/classification attributes
  SCENE_TEXT, LICENSE_PLATE, FACE_DETECTION, PEDESTRIAN_DETECTION, 
  FACE_RECOGNITION, BODY_POSE, GENDER, AGE, EXPRESSION, 
  GRAPHIC_CONTENT, INSIGNIA, MUNITION, FIREARM, LOGO
  """
  # computed: MEDIAINFO, EXIF, KEYFRAME, KEYFRAME_STATUS
  # lowercase enum names correspond exactly to filesystem directories
  # order doesn't matter and change but labels must never change

  MEDIA_RECORD, SUGARCUBE, MEDIAINFO, EXIF, KEYFRAME, KEYFRAME_STATUS, \
  FEATURE_VGG16, FEATURE_RESNET18, FEATURE_ALEXNET, VOC, COCO, \
  PLACES365, OPENIMAGES, SUBMUNITION = range(14)


class MetadataTree(Enum):
  """Represents the (deprecated) metadata tree structures"""
  MEDIAINFO_TREE, KEYFRAME_TREE = range(2)

class MediaRecord(Enum): 
  """Type of item records (for now everything is SHA256-based"""
  SHA256 = 1

class ClientRecord(Enum): 
  """Type of item records. for now only one type"""
  SUGARCUBE = 1

class MediainfoMetadata(Enum):
  """Types of mediainfo metadata"""
  AUDIO, VIDEO = range(2)

class KeyframeMetadata(Enum):
  """Types of keyframes available"""
  DENSE, BASIC, EXPANDED = range(3)


# -------------------------------------------------------------------
# Object detection and classification networks

class DarknetClassify(Enum):
  """Darknet networks"""
  IMAGENET = 1

class DarknetDetect(Enum):
  """Darknet networks"""
  COCO, COCO_SPP, VOC, OPENIMAGES, SUBMUNITION = range(5)

class KerasNet(Enum):
  """Keras weights for feature extractor"""
  DENSENET121, DENSENET160, DENSENET169, DENSENET201, \
    INCEPTIONV2, INCEPTIONV3, NASNETLARGE, \
    NASNETMOBILE, RESNET50, INCEPTIONRESNETV2, \
    VGG16, VGG19, XCEPTION = range(13)

class PyTorchNet(Enum):
  """Types of PyTorch weights for feature extractor"""
  ALEXNET, RESNET18 = range(2)

# ---------------------------------------------------------------------
# Status 
# --------------------------------------------------------------------

class SearchParam(Enum):
  """Parameters used for looking up an ID"""
  SHA256, MD5, SA_ID = range(3)

class Verified(Enum):
  UNVERIFIED, VERIFIED = range(2)

class KeyframeStatus(Enum):
  """Keyframe image extraction status"""
  VALID, INVALID, UNSET = range(3)

class MetadataStatus(Enum):
  """Metadata analysis status"""
  INVALID, VALID, UNSET = range(3)


# ---------------------------------------------------------------------
# File Types and Paths
# --------------------------------------------------------------------
class FileExt(Enum):
  JSON, PKL = range(2)

class MediaFormat(Enum):
  """Media type to be analyzed"""
  VIDEO, KEYFRAME, PHOTO = range(3)

class DataStore(Enum):
  """Storage devices. Paths are symlinked to root (eg /data_store_nas)"""
  NAS, HDD, SSD = range(3)

# ---------------------------------------------------------------------
# Image processing
# --------------------------------------------------------------------

# Image sizes for saving
class ImageSize(Enum):
  """Image sizes used for keyframes and resized photos"""
  THUMB, SMALL, MEDIUM, LARGE = range(4)

class CVBackend(Enum):
  """OpenCV 3.4.2+ DNN target type"""
  DEFAULT, HALIDE, INFER_ENGINE, OPENCV = range(4)

class CVTarget(Enum):
  """OpenCV 3.4.2+ DNN backend processor type"""
  CPU, OPENCL, OPENCL_FP16, MYRIAD = range(4)

class VideoQuality(Enum):
  """Describes video quality by size, frame rate"""
  POOR, LOW, MEDIUM, HIGH, HD = range(5)


# ---------------------------------------------------------------------
# Logger, monitoring
# --------------------------------------------------------------------

class LogLevel(Enum):
  """Loger vebosity"""
  DEBUG, INFO, WARN, ERROR, CRITICAL = range(5)






# not in use
class ItemGenerator(Enum):
  """Type of file to use for generator item source"""
  MAPPING, METADATA = range(2)
