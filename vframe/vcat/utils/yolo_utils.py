from os.path import join
import csv

from vframe.utils import file_utils, logger_utils
from vframe.settings import vframe_cfg as cfg
from vcat.settings import vcat_cfg
from vcat.models.yolo_item import YoloAnnoItem
from vcat.utils import vcat_utils


log = logger_utils.Logger.getLogger()

# ---------------------------------------------------    
# Creates list of yolo objects w/relevant annotation data
# ---------------------------------------------------

def gen_readme():
  """Generates a README.md summary file for the project"""
  return ''

# ---------------------------------------------------    
# Creates list of yolo objects w/relevant annotation data
# ---------------------------------------------------

def gen_annos(vcat_data, parent_hierarchy=False):
  """Generates list of YoloAnnoItem objects"""

  log.debug('generate annotations')

  # get the ordered hierarchy
  hierarchy_tree = vcat_utils.hierarchy_tree(vcat_data['hierarchy'].copy())
  annos_tree = vcat_utils.hierarchy_tree_display(hierarchy_tree)
  anno_idx_lookup = vcat_utils.hierarchy_flat(hierarchy_tree)

  # if opt, append parent clases
  if parent_hierarchy:
    vcat_utils.append_parents(anno_idx_lookup, vcat_data['hierarchy'])

  # create a lookup table for slugs --> class index
  slug_lookup = {v['slug']: k for k, v in anno_idx_lookup.items()}

  # write labels files for all annos
  annos = []

  # build image ID lookup table. the regions refer to these
  image_lookup =  {}
  for class_id, object_class in vcat_data['object_classes'].items():
    for image in object_class['images']:
      image_lookup[image['id']] = image
    

  for class_id, object_class in vcat_data['object_classes'].items():
    log.debug('class id: {}: {}'.format(class_id, object_class['slug']))
    class_id = slug_lookup[object_class['slug']]

    for region in object_class['regions']:
      image = image_lookup[region['image']]
      fn = image['fn']
      yai = YoloAnnoItem(fn, class_id, region)
      annos.append(yai)

      if parent_hierarchy and bool(object_class['is_attribute']):
        # get parent id
        parent_vcat_id = int(object_class['parent'])
        # get parent hierarchy object
        parent_obj = vcat_data['hierarchy'][parent_vcat_id]
        # use slug to get class label id
        parent_slug = parent_obj['slug']
        parent_class_id = slug_lookup[parent_slug]
        yai = YoloAnnoItem(fn, parent_class_id, region)
        annos.append(yai)

  return annos



# ---------------------------------------------------    
# Create "yolov3.cfg"
# ---------------------------------------------------

def gen_cfg(cfg_type, n_classes, n_batch, n_subdiv, opt_size):
  """Generates configuration file from yolo templates"""
  
  cfg_orig = open(vcat_cfg.FP_YOLOV3_CFG,'r')    

  num_filters = (n_classes + 5) * 3
  s_replace_base = [
    ('filters=255', 'filters={}'.format(num_filters)),
    ('classes=80', 'classes={}'.format(n_classes)),
    ]

  if cfg_type == 'train':
    s_replace = s_replace_base.copy()
    s_replace.append(('batch=1','#batch=1'))
    s_replace.append(('subdivisions=1\n','#subdivisions=1\n'))
    s_replace.append(('# batch=64','batch={}'.format(n_batch)))
    s_replace.append(('# subdivisions=16','subdivisions={}'.format(n_subdiv)))
  elif cfg_type == 'test':
    # change nclasses, but keep the subdiv and batch size at default
    s_replace = s_replace_base.copy() 

  # change size
  s_replace.append(('width=416','width={}'.format(opt_size[0])))
  s_replace.append(('height=416','height={}'.format(opt_size[1])))
  
  # search and replace, return list of lines
  cfg_new_data = []
  for line in cfg_orig:
    for query,targ in s_replace:
      if query in line:
        line = line.replace(query,targ)
    cfg_new_data.append(line.replace('\n',''))

  return cfg_new_data


# ---------------------------------------------------    
# Create shell scripts to star training
# ---------------------------------------------------
def gen_sh_train(dir_project, fp_meta, fp_cfg, opt_gpus, weights=None):
  """Generates train and resume shell scripts"""

  txts = []
  txts.append('#!/bin/bash')
  txts.append('DARKNET={}'.format(vcat_cfg.FP_DARKNET_BIN))
  txts.append('DIR_PROJECT={}'.format(dir_project))
  txts.append('FP_META={}'.format(fp_meta))
  txts.append('FP_CFG={}'.format(fp_cfg))
  txts.append('CMD="detector train"')
  if not weights:
    txts.append('FP_WEIGHTS=weights/yolov3.backup')
  else:
    txts.append('FP_WEIGHTS={}'.format(weights))  
  txts.append('GPUS="-gpus {}"'.format(opt_gpus))
  txts.append('$DARKNET $CMD $FP_META $FP_CFG $FP_WEIGHTS $GPUS 2>&1 | tee training.log')
  return txts


def gen_sh_test(dir_project, fp_meta, fp_cfg_test):
  """Generates a shell script for testing"""
  txts = []
  txts.append('#!/bin/bash')
  txts.append('DARKNET={}'.format(vcat_cfg.FP_DARKNET_BIN))
  txts.append('DIR_PROJECT={}'.format(dir_project))
  txts.append('FP_CFG={}'.format(fp_cfg_test))
  txts.append('FP_META={}'.format(fp_meta))
  txts.append('#FP_WEIGHTS=weights/{}_900.weights'.format('yolov3'))
  txts.append('FP_WEIGHTS=weights/{}_10000.weights'.format('yolov3'))
  txts.append('#FP_WEIGHTS=weights/{}_30000.weights'.format('yolov3'))
  txts.append('CMD="detector test"')
  # txts.append('cd $DARKNET')
  txts.append('$DARKNET $CMD $FP_META $FP_CFG $FP_WEIGHTS $1')
  return txts

# ---------------------------------------------------    
# Create "meta.data" text
# ---------------------------------------------------
def gen_meta(dir_project, num_classes):
  """Generates the .data file for training"""
  data = []
  data.append('classes = {}'.format(num_classes))
  data.append('train = {}'.format(join(dir_project, 'train.txt')))
  data.append('valid = {}'.format(join(dir_project, 'valid.txt')))
  data.append('names = {}'.format(join(dir_project, 'classes.txt')))
  data.append('backup = {}'.format(join(dir_project, 'weights/')))
  return data 