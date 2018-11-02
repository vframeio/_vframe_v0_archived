import sys
import os
from os.path import join
import json
import requests

from vframe.settings import vframe_cfg as cfg
from vcat.settings import vcat_cfg
from vframe.utils import logger_utils

# --------------------------------------------------------
# Downloads data from VCAT API
# --------------------------------------------------------

class API:

  def __init__(self, un, pw):
    self.log = logger_utils.Logger.getLogger()
  
    if not un or not pw:
      self.log.error('Username and/or password not supplied')
      sys.exit()
    self.un = un
    self.pw = pw
    # TODO move to config
    self.hierarchy_url = vcat_cfg.VCAT_HIERARCHY_URL

  def get_hierarchy(self):
    try:
      hierarchy_raw = requests.get(self.hierarchy_url, auth=(self.un, self.pw)).json()
    except:
      self.log.error('Could not get data from: {}'.format(self.hierarchy_url))
      return {}

    return { int(class_meta['id']): class_meta for class_meta in hierarchy_raw }


  def request_regions(self, class_id):
    url = join(self.hierarchy_url, class_id, 'regions')
    return requests.get(url,auth=(self.un, self.pw)).json()


  def get_class(self, class_id):
    """get single class, but with same format"""
    object_classes = { class_id: self.request_regions(class_id) }
    return {'object_classes': object_classes}


  def get_full(self):

    objs_regions = {}
    hierarchy = self.get_hierarchy()
    for hierarchy_id, hierarchy_obj in hierarchy.items():
      if int(hierarchy_obj['region_count']) > 0:
        slug = hierarchy_obj['slug'].replace(':','').replace('-','_')
        cat_id = str(hierarchy_obj['id'])
        obj_regions = self.request_regions(cat_id)
        objs_regions[cat_id] = obj_regions

    return {'hierarchy':hierarchy, 'object_classes':objs_regions}

