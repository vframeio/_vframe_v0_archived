import os
from os.path import join
import logging

from vframe.settings import vframe_cfg as vcfg
from vframe.settings import types

class Paths:
  
  # class properties
  MAPPINGS_DATE = vcfg.SUGARCUBE_DATES[0]
  DIR_APP_VFRAME = 'apps/vframe/'
  DIR_APP_SA = 'apps/syrianarchive'
  DIR_MODELS_VFRAME = join(DIR_APP_VFRAME, 'models')
  DIR_DARKNET = join(DIR_MODELS_VFRAME, 'darknet/pjreddie')
  DIR_DARKNET_VFRAME = join(DIR_MODELS_VFRAME, 'darknet/vframe')
  DIR_MEDIA = join(DIR_APP_SA, 'media')
  DIR_METADATA = join(DIR_APP_SA, 'metadata')
  DIR_RECORDS = join(DIR_APP_SA, 'records')
  DIR_REPORTS = join(DIR_APP_SA, 'reports')


  def __init__(self):
    pass

  @classmethod
  def DataStorePath(cls, data_store=types.DataStore.HDD):
    return '/data_store_{}'.format(data_store.name.lower())

  # -------------------------------------------------------------------------------
  # Darknet Paths

  @classmethod
  def darknet_classes(cls, data_store=types.DataStore.HDD, opt_net=types.DarknetDetect.COCO):
    if opt_net == types.DarknetDetect.COCO:
      fp = join(cls.DIR_DARKNET, 'data', 'coco.names')
    elif opt_net == types.DarknetDetect.COCO_SPP:
      fp = join(cls.DIR_DARKNET, 'data', 'coco.names')
    elif opt_net == types.DarknetDetect.VOC:
      fp = join(cls.DIR_DARKNET, 'data', 'voc.names')
    elif opt_net == types.DarknetDetect.OPENIMAGES:
      fp = join(cls.DIR_DARKNET, 'data', 'openimages.names')
    elif opt_net == types.DarknetDetect.SUBMUNITION:
      fp = join(cls.DIR_DARKNET_VFRAME, 'cluster_munition_07', 'classes.txt')
    return join(cls.DataStorePath(data_store), fp)

  @classmethod
  def darknet_data(cls, data_store=types.DataStore.HDD, opt_net=types.DarknetDetect.COCO):
    if opt_net == types.DarknetDetect.COCO:
      fp = join(cls.DIR_DARKNET, 'cfg', 'coco.data')
    elif opt_net == types.DarknetDetect.COCO_SPP:
      fp = join(cls.DIR_DARKNET, 'cfg', 'coco.data')
    elif opt_net == types.DarknetDetect.VOC:
      fp = join(cls.DIR_DARKNET, 'cfg', 'voc.data')
    elif opt_net == types.DarknetDetect.OPENIMAGES:
      fp = join(cls.DIR_DARKNET, 'cfg', 'openimages.data')
    elif opt_net == types.DarknetDetect.SUBMUNITION:
      fp = join(cls.DIR_DARKNET_VFRAME, 'cluster_munition_07', 'meta.data')
    fp = join(cls.DataStorePath(data_store), fp)
    return bytes(fp, encoding="utf-8")

  @classmethod
  def darknet_cfg(cls, data_store=types.DataStore.HDD, opt_net=types.DarknetDetect.COCO):
    if opt_net == types.DarknetDetect.COCO:
      fp = join(cls.DIR_DARKNET, 'cfg', 'yolov3.cfg')
    elif opt_net == types.DarknetDetect.COCO_SPP:
      fp = join(cls.DIR_DARKNET, 'cfg', 'yolov3-spp.cfg')
    elif opt_net == types.DarknetDetect.VOC:
      fp = join(cls.DIR_DARKNET, 'cfg', 'yolov3-voc.cfg')
    elif opt_net == types.DarknetDetect.OPENIMAGES:
      fp = join(cls.DIR_DARKNET, 'cfg', 'yolov3-openimages.cfg')
    elif opt_net == types.DarknetDetect.SUBMUNITION:
      fp = join(cls.DIR_DARKNET_VFRAME, 'cluster_munition_07', 'yolov3.cfg')
    fp = join(cls.DataStorePath(data_store), fp)
    return bytes(fp, encoding="utf-8")

  @classmethod
  def darknet_weights(cls, data_store=types.DataStore.HDD, opt_net=types.DarknetDetect.COCO):
    if opt_net == types.DarknetDetect.COCO:
      fp = join(cls.DIR_DARKNET, 'weights', 'yolov3.weights')
    elif opt_net == types.DarknetDetect.COCO_SPP:
      fp = join(cls.DIR_DARKNET, 'weights', 'yolov3-spp.weights')
    elif opt_net == types.DarknetDetect.VOC:
      fp = join(cls.DIR_DARKNET, 'weights', 'yolov3-voc.weights')
    elif opt_net == types.DarknetDetect.OPENIMAGES:
      fp = join(cls.DIR_DARKNET, 'weights', 'yolov3-openimages.weights')
    elif opt_net == types.DarknetDetect.SUBMUNITION:
      fp = join(cls.DIR_DARKNET_VFRAME, 'cluster_munition_07/weights', 'yolov3_11000.weights')
    fp = join(cls.DataStorePath(data_store), fp)
    return bytes(fp, encoding="utf-8")

  # -------------------------------------------------------------------------------
  # Metadata Paths

  @classmethod
  def mapping_index(cls, opt_date, data_store=types.DataStore.HDD, verified=types.Verified.VERIFIED, 
    file_format=types.FileExt.PKL):
    """Returns filepath to a mapping file. Mapping files are the original Suguarcube mapping data"""
    fname = 'index.pkl' if file_format == types.FileExt.PKL else 'index.json'
    # data_store = 'data_store_{}'.format(data_store.name.lower())
    date_str = opt_date.name.lower()
    fp = join(cls.DataStorePath(data_store), cls.DIR_METADATA, 'mapping', date_str, verified.name.lower(), fname)
    return fp

  @classmethod
  def media_record_index(cls, data_store=types.DataStore.HDD, verified=types.Verified.VERIFIED, 
    file_format=types.FileExt.PKL):
    """Returns filepath to a mapping file. Mapping files are the original Suguarcube mapping data"""
    fname = 'index.pkl' if file_format == types.FileExt.PKL else 'index.json'
    metadata_type = types.Metadata.MEDIA_RECORD.name.lower()
    fp = join(cls.DataStorePath(data_store), cls.DIR_METADATA, metadata_type, verified.name.lower(), fname)
    return fp

  @classmethod
  def metadata_index(cls, metadata_type, data_store=types.DataStore.HDD, 
    verified=types.Verified.VERIFIED, file_format=types.FileExt.PKL):
    """Uses key from enum to get folder name and construct filepath"""
    fname = 'index.pkl' if file_format == types.FileExt.PKL else 'index.json'
    fp = join(cls.DataStorePath(data_store), cls.DIR_METADATA, metadata_type.name.lower(), 
      verified.name.lower(), fname)
    return fp

  @classmethod
  def metadata_dir(cls, metadata_type, data_store=types.DataStore.HDD, verified=types.Verified.VERIFIED):
    """Uses key from enum to get folder name and construct filepath"""
    fp = join(cls.DataStorePath(data_store), cls.DIR_METADATA, metadata_type.name.lower(), 
      verified.name.lower())
    return fp

  @classmethod
  def metadata_tree_dir(cls, metadata_type, data_store=types.DataStore.HDD):
    """Uses key from enum to get folder name and construct filepath"""
    fp = join(cls.DataStorePath(data_store), cls.DIR_METADATA, metadata_type.name.lower())
    return fp

  @classmethod
  def media_dir(cls, media_type, data_store=types.DataStore.HDD, verified=types.Verified.VERIFIED):
    """Returns the directory path to a media directory"""
    fp = join(cls.DataStorePath(data_store), cls.DIR_MEDIA, media_type.name.lower(), verified.name.lower())
    return fp

  # @classmethod
  # def keyframe(cls, dir_media, idx, image_size=types.ImageSize.MEDIUM):
  #   """Returns path to keyframe image using supplied cls.media directory"""
  #   idx = str(idx).zfill(vcfg.ZERO_PADDING)
  #   size_label = vcfg.IMAGE_SIZE_LABELS[image_size]
  #   fp = join(dir_media, sha256_tree, sha256, idx, size_label, 'index.jpg')
  #   return fp

  @classmethod
  def dnn(cls):
    """Returns configurations for available DNNs"""
    pass