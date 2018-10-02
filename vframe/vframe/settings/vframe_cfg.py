import os
from os.path import join
import logging
import collections

import cv2 as cv

from vframe.settings import types
from vframe.models.video_item import VideoQuality
from vframe.utils import click_utils

# -----------------------------------------------------------------------------
# Enun lists used for custom Click Params
# -----------------------------------------------------------------------------

DarknetClassifyVar = click_utils.ParamVar(types.DarknetClassify)
DarknetDetectVar = click_utils.ParamVar(types.DarknetDetect)
PyTorchNetVar = click_utils.ParamVar(types.PyTorchNet)
KerasNetVar = click_utils.ParamVar(types.KerasNet)

SearchParamVar = click_utils.ParamVar(types.SearchParam)
ClientRecordVar = click_utils.ParamVar(types.ClientRecord)
MetadataTreeVar = click_utils.ParamVar(types.MetadataTree)
ImageSizeVar = click_utils.ParamVar(types.ImageSize)

VideoQualityVar = click_utils.ParamVar(types.VideoQuality)
DataStoreVar = click_utils.ParamVar(types.DataStore)
FileExtVar = click_utils.ParamVar(types.FileExt)

KeyframeMetadataVar = click_utils.ParamVar(types.KeyframeMetadata)
MediaRecordVar = click_utils.ParamVar(types.MediaRecord)
VerifiedVar = click_utils.ParamVar(types.Verified)
MediaFormatVar = click_utils.ParamVar(types.MediaFormat)
MetadataVar = click_utils.ParamVar(types.Metadata)
LogLevelVar = click_utils.ParamVar(types.LogLevel)



# # data_store
DATA_STORE = '/data_store/'
DIR_DATASETS = join(DATA_STORE,'datasets')
DIR_APPS = join(DATA_STORE,'apps')
DIR_APP_VFRAME = join(DIR_APPS,'vframe')
DIR_APP_SA = join(DIR_APP_VFRAME, 'syrianarchive')
DIR_MODELS_VFRAME = join(DIR_APP_VFRAME,'models')
DIR_MODELS_SA = join(DIR_APP_SA,'models')

# # Frameworks
DIR_MODELS_OPENCV = join(DIR_MODELS_VFRAME,'caffe')
DIR_MODELS_CAFFE = join(DIR_MODELS_VFRAME,'caffe')
DIR_MODELS_DARKNET = join(DIR_MODELS_VFRAME,'darknet')
DIR_MODELS_DARKNET_PJREDDIE = join(DIR_MODELS_DARKNET, 'pjreddie')
DIR_MODELS_DARKNET_VFRAME = join(DIR_MODELS_DARKNET, 'vframe')
DIR_MODELS_PYTORCH = join(DIR_MODELS_VFRAME,'pytorch')
DIR_MODELS_TORCH = join(DIR_MODELS_VFRAME,'torch')
DIR_MODELS_MXNET = join(DIR_MODELS_VFRAME,'mxnet')
DIR_MODELS_TF = join(DIR_MODELS_VFRAME,'tensorflow')


# -----------------------------------------------------------------------------
# click chair settings
# -----------------------------------------------------------------------------
DIR_COMMANDS_PROCESSOR_CHAIR = 'vframe/commands/'
DIR_COMMANDS_PROCESSOR_VCAT = 'vcat/commands/'
DIR_COMMANDS_PROCESSOR_ADMIN = 'admin/commands'

# -----------------------------------------------------------------------------
# Sugarcube dates
# Dates the snaphots are made
# -----------------------------------------------------------------------------
SUGARCUBE_DATES = ['20180611']

# -----------------------------------------------------------------------------
# Filesystem settings
# hash trees enforce a maximum number of directories per directory
# -----------------------------------------------------------------------------
ZERO_PADDING = 6  # padding for enumerated image filenames
FRAME_NAME_ZERO_PADDING = 6  # is this active??
HASH_TREE_DEPTH = 3
HASH_BRANCH_SIZE = 3

# -----------------------------------------------------------------------------
# Logging options exposed for custom click Params
# -----------------------------------------------------------------------------
LOGGER_NAME = 'vframe'
LOGLEVELS = {
  types.LogLevel.DEBUG: logging.DEBUG,
  types.LogLevel.INFO: logging.INFO,
  types.LogLevel.WARN: logging.WARN,
  types.LogLevel.ERROR: logging.ERROR,
  types.LogLevel.CRITICAL: logging.CRITICAL
}
LOGLEVEL_OPT_DEFAULT = types.LogLevel.DEBUG.name
#LOGFILE_FORMAT = "%(asctime)s: %(levelname)s: %(message)s"
#LOGFILE_FORMAT = "%(levelname)s:%(name)s: %(message)s"
#LOGFILE_FORMAT = "%(levelname)s: %(message)s"
#LOGFILE_FORMAT = "%(filename)s:%(lineno)s  %(funcName)s()  %(message)s"
# colored logs
"""
black, red, green, yellow, blue, purple, cyan and white.
{color}, fg_{color}, bg_{color}: Foreground and background colors.
bold, bold_{color}, fg_bold_{color}, bg_bold_{color}: Bold/bright colors.
reset: Clear all formatting (both foreground and background colors).
"""
LOGFILE_FORMAT = "%(log_color)s%(levelname)-8s%(reset)s %(cyan)s%(filename)s:%(lineno)s:%(bold_cyan)s%(funcName)s() %(reset)s%(message)s"

# -----------------------------------------------------------------------------
# Media formats accepted by VFRAME
# -----------------------------------------------------------------------------
VALID_MEDIA_EXTS = {
  types.MediaFormat.VIDEO: ['mp4','mov','avi'],
  types.MediaFormat.PHOTO: ['jpg','jpeg','png']
}

# -----------------------------------------------------------------------------
# Image size for web images
# -----------------------------------------------------------------------------
# order here is used for effecient image-pyramid resizing
IMAGE_SIZES = collections.OrderedDict()
IMAGE_SIZES[types.ImageSize.THUMB] = 160
IMAGE_SIZES[types.ImageSize.SMALL] = 320
IMAGE_SIZES[types.ImageSize.MEDIUM] = 640
IMAGE_SIZES[types.ImageSize.LARGE] = 1280

IMAGE_SIZE_LABELS = collections.OrderedDict()
IMAGE_SIZE_LABELS[types.ImageSize.THUMB] = 'th'
IMAGE_SIZE_LABELS[types.ImageSize.SMALL] = 'sm'
IMAGE_SIZE_LABELS[types.ImageSize.MEDIUM] = 'md'
IMAGE_SIZE_LABELS[types.ImageSize.LARGE] = 'lg'
DEFAULT_SIZE_LABEL_FEAT_EXTRACT = IMAGE_SIZE_LABELS[types.ImageSize.MEDIUM]
JPG_SAVE_QUALITY = 75
KEYFRAME_EXT = 'jpg'


# Define video quality metrics (w, h, fps, sec)
VIDEO_QUALITY = collections.OrderedDict()
VIDEO_QUALITY[types.VideoQuality.POOR] = VideoQuality(160, 90, 12, 2)
VIDEO_QUALITY[types.VideoQuality.LOW] = VideoQuality(320, 180, 12, 2)
VIDEO_QUALITY[types.VideoQuality.MEDIUM] = VideoQuality(640, 360, 12, 2)
VIDEO_QUALITY[types.VideoQuality.HIGH] = VideoQuality(1280, 720, 12, 2)  # HD Ready
VIDEO_QUALITY[types.VideoQuality.HD] = VideoQuality(1920, 1080, 24, 2)  # Full HD

# -----------------------------------------------------------------------------
# OpenCV backend and target
# used for optimizing DNN inference speeds
# requires OpenCV >= 3.4.2
# -----------------------------------------------------------------------------
OPENCV_DNN_BACKENDS = {
  types.CVBackend.DEFAULT: cv.dnn.DNN_BACKEND_DEFAULT, 
  types.CVBackend.HALIDE: cv.dnn.DNN_BACKEND_HALIDE,
  types.CVBackend.INFER_ENGINE: cv.dnn.DNN_BACKEND_INFERENCE_ENGINE,
  types.CVBackend.OPENCV: cv.dnn.DNN_BACKEND_OPENCV
}
OPENCV_DNN_TARGETS = {
  types.CVTarget.CPU: cv.dnn.DNN_TARGET_CPU, 
  types.CVTarget.OPENCL: cv.dnn.DNN_TARGET_OPENCL,
  types.CVTarget.OPENCL_FP16: cv.dnn.DNN_TARGET_OPENCL_FP16,
  types.CVTarget.MYRIAD: cv.dnn.DNN_TARGET_MYRIAD
}
OPENCV_BACKEND_DEFAULT = types.CVBackend.OPENCV
OPENCV_TARGET_DEFAULT = types.CVTarget.OPENCL_FP16


# -----------------------------------------------------------------------------
# Minimum keyframe extraction video attributes
# -----------------------------------------------------------------------------
KEYFRAME_MIN_WIDTH = 640
#KEYFRAME_MIN_WIDTH = 480  # some verified videos are this small, ignore for now
KEYFRAME_MIN_HEIGHT = 320
KEYFRAME_MIN_FPS = 10
KEYFRAME_MIN_FRAMES = 90


# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------


    
