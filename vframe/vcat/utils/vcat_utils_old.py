import os
from os.path import join
import sys
import json
import numpy as np
import cv2 as cv
import click
import requests
import imutils
from tqdm import tqdm
import random
import collections
from operator import itemgetter
# local
sys.path.append('/vframe/tools')
from utils import fiox


class VcatUtils:

  def __init__(self):
    pass

  def load_data(self, fp, exclude=None):
    """
    Loads VCAT data and exludes classes if given
    :param fp: click.File object
    :param exclude: string of classes to exclude
    :returns: updated complete VCAT JSON object
    """
    # load vcat JSON
    vcat_data = json.load(fp)

    # exclude classes
    if exclude is not None:
      excludes = [str(i) for i in exclude.split(',')]
      vcat_data = self.exclude_classes(vcat_data, excludes)
    return vcat_data

  def exclude_classes(self, vcat_data, excludes):
    """
    Remove classes from VCAT hierarchy and object_classes
    :param vcat_cata: full VCAT data in JSON format
    :param exclusions: integer-list of exlcluded classes
    :returns: revised VCAT data in JSZON format
    """

    print('[+] Excluding: {}'.format(excludes))

    # remove from hierarchy
    hierarchy = vcat_data['hierarchy']
    hierarchy_tmp = hierarchy.copy()
    for class_id, class_meta in hierarchy_tmp.items():
      if class_id in excludes:
        print('[+] Removing hierarchy ID: {} ({})'.format(class_id, class_meta['slug']))
        del hierarchy[class_id]

    # remove from annotations
    object_classes = vcat_data['object_classes']
    object_classes_tmp = object_classes.copy()
    for class_id, class_meta in object_classes_tmp.items():
      if class_id in excludes:
        print('[+] Removing annotation ID: {} ({})'.format(class_id, class_meta['slug']))
        del hierarchy[class_id]

    return {'hierarchy': hierarchy, 'object_classes': object_classes}


  def create_splits(self,anno_index,randomize=True,split_val=0.2,rseed=1):
    """Convert annotation index into splits based on classes"""
    
    # TODO add num_split_vals
    annos_train = {}
    annos_valid = {}
    for class_idx, anno_obj in anno_index.items():
      n_annos = len(anno_obj['regions'])
      if randomize:
        random.seed(rseed)
        #random.shuffle(anno_obj['regions'])
      n_test = int(n_annos*split_val)
      class_annos_valid = anno_obj.copy()
      class_annos_valid['regions'] = anno_obj['regions'][:n_test]
      class_annos_train = anno_obj.copy()
      class_annos_train['regions'] = anno_obj['regions'][n_test:]
      annos_train[class_idx] = class_annos_train
      annos_valid[class_idx] = class_annos_valid

    return {'train':annos_train,'valid':annos_valid}


  def create_im_index(self,data):
    """Return a lookup table for images used in annotations"""
    hierarchy = data['hierarchy']
    classes = data['object_classes']
    im_id_index = {}
    for obj_class_id, obj_class in hierarchy.items():
      if int(obj_class['region_count'] > 0):
        obj_images = classes[str(obj_class['id'])]['images']
        for obj_image in obj_images:
          im_id = int(obj_image['id'])
          if im_id not in im_id_index.keys():
            im_id_index[im_id] = {'fn':obj_image['fn'],'ext':obj_image['ext']}
    return im_id_index


  def hierarchy_tree(self, hierarchy):
    """Convert VCAT flat hierarchy to a tree sturcture"""

    tree = {}

    # get root level
    for class_id, class_meta in hierarchy.copy().items():
      class_id = str(class_id)
      parent_id = class_meta['parent']

      if parent_id is None:
        # top level
        tree[class_id] = class_meta
        tree[class_id]['subclasses'] = {}
        tree[class_id]['attributes'] = {}
        tree[class_id]['label_index'] = len(tree) - 1
        hierarchy.pop(class_id)
    
    # recursive add subclasses
    def add_subclasses(tree, hierarchy, index=0):
      # for each class ID at this level
      for tree_id, tree_meta in tree.items():
        # add hierarchy items if parent matches class
        tree_id = str(tree_id)
        for class_id, class_meta in hierarchy.copy().items():
          # add child to parent class
          class_id = str(class_id)
          parent_id = str(class_meta['parent'])
          if parent_id == tree_id:
            # consume hierarchy elements
            hierarchy.pop(class_id)
            # add attribute if exists
            if class_meta['is_attribute']:
              class_meta['label_index'] = index
              index += 1
              tree_meta['attributes'][class_id] = class_meta
            else:
              # add subclass
              class_meta['label_index'] = index
              index += 1
              class_meta['subclasses'] = {}
              class_meta['attributes'] = {}
              tree[tree_id]['subclasses'][class_id] = class_meta
        # find all subclasses and attributes of this tree
        tree[tree_id]['subclasses'] = add_subclasses(tree_meta['subclasses'], hierarchy, index=index)
      return tree

    tree = add_subclasses(tree, hierarchy, index=len(tree))    
    return tree
  

  def hierarchy_flat(self, tree, sort=True):
    def walk_tree(tree, flat={}):
      for class_id, class_meta in tree.items():
        subclasses = class_meta.get('subclasses',None)
        flat[class_meta['label_index']] = class_meta
        flat = walk_tree(class_meta['subclasses'], flat)
        for attr_id, attr_meta in class_meta['attributes'].items():
          flat[attr_meta['label_index']] = attr_meta
      return flat
    flat = walk_tree(tree)
    if sort:
      flat = collections.OrderedDict(sorted(flat.items()))
    return flat

  def display_hierarchy(self, tree, output=[], indent=0):
    """Returns class hierarchy in space formatted style"""
    for k,v in tree.items():
      output.append('{}+ {} (VCAT ID: {}, Training ID: {})'.format(indent*'  ', v['slug'], v['id'], v['label_index']))
      subclasses = v.get('subclasses',None)
      self.display_hierarchy(subclasses, output=output, indent=indent +1)
      attributes = v.get('attributes',None)
      for attr_id, attr_meta in attributes.items():
       # output.append('{} - [{}] {}'.format(indent*'  ',attr_meta['label_index'],attr_meta['slug']))
       output.append('{} - {} (VCAT ID: {}, Training ID: {}'.format(indent*'  ', attr_meta['slug'], attr_meta['id'], attr_meta['label_index']))
    return output

  def format_labels(self,format_type):
    if format_type == 'yolo':
      pass
    elif format_type == 'autokeras':
      pass


  def create_class_label_index(self,vcat_hierarchy,parents=False):
    """Build lookup table for class label index"""

    #class_labels = []
    hierarchy = {}

    for class_id, class_meta in vcat_hierarchy.items():

      is_attribute = class_meta['is_attribute']
      has_regions = int(class_meta['region_count']) > 0

      if not has_regions: continue
      
      # ensure parent node
      if parents:
        parent_id = class_meta['parent']
        if parent_id not in hierarchy.keys():
          parent_meta = vcat_hierarchy[parent_id]
          parent_meta['subclasses'] = []
          hierarchy[parent_id] = parent_meta
        
      if is_attribute:
        
        
        #hierarchy[parent_id]['subclasses'].append(class)
        hierarchy[parent_id]  
      # add object class
      class_ids[class_id] = class_meta['parent']
        
    # sort this list based on slug and add key for class label index
    #class_ids.sort(key=itemgetter('id'))
    return hierarchy

    # return lookup table
    class_label_index = {}
    count = 0
    for v in class_labels:
      if v['id'] not in class_label_index.keys():
        class_label_index[v['id']] = count # vcat ID --> yolo ID
        count += 1
    
    class_label_index = collections.OrderedDict(sorted(class_label_index.items()))
    return class_label_index


  def create_annotation_index(self,vcat_data,class_label_index_lkup,kwargs):
    """
    Convert JSON file of annotations to images

    param: annos: JSON object containing bboxes and local image path
    returns: list of dicts with image and class name
    """

    # parse vcat JS1ON
    hierarchy = vcat_data['hierarchy']
    object_classes = vcat_data['object_classes']

    # create index-based lookup table of filenames
    im_id_index = self.create_im_index(vcat_data)

    # store object class annotation objects here
    anno_index_dict = {}

    for obj_id_key, obj_meta in hierarchy.items():

      # skip if no annotations
      if not int(obj_meta['region_count'] > 0): continue
      
      # get current object info
      obj_class_id = int(obj_meta['id'])
      slug_vcat = str(obj_meta['slug']) # vcat slug style
      slug_local = slug_vcat.replace(':','_').replace('-','_') # only uderscores
      anno_meta = object_classes[str(obj_class_id)]
      
      # copy object info
      obj_meta = anno_meta.copy()

      # update object info
      obj_meta['slug_local'] = slug_local
      obj_meta['slug_vcat'] = slug_vcat
      obj_meta['class_id'] = obj_class_id
      class_label_index = class_label_index_lkup[obj_class_id]
      obj_meta['class_label_index'] = class_label_index
        
      # copy the region/annoations to the object metadata
      obj_meta['regions'] = object_classes[str(obj_class_id)]['regions'].copy()
      #obj_meta['is_attribute'] = object_classes[str(obj_class_id)]['is_attribute']

      # add the local filesystem path to the object metadata
      for region in obj_meta['regions']:
        
        im_id = int(region['image'])
        fname = im_id_index[im_id]['fn']
        fext = im_id_index[im_id]['ext']
        region['fn'] = fname
        region['ext'] = fext
        region['ext'] = fext
        is_attribute = bool(obj_meta['is_attribute'])
        region['is_attribute'] = is_attribute
        region['class_label_index'] = class_label_index
        
        if kwargs['parent'] and is_attribute:
          # add parent id --> class label index
          parent_id = int(obj_meta['parent'])
          # print('add parent {}'.format(parent_id))
          region['parent_class_label_index'] = class_label_index_lkup[parent_id]

        region['filepath'] = os.path.join(kwargs['dir_images'],'{}/{}/{}_lg{}'.format(
          im_id,fname,fname,fext))
        region['basename'] = os.path.basename(region['filepath'])
        
      #anno_index.append(obj_meta)
      anno_index_dict[class_label_index] = obj_meta


    # sort
    anno_index_dict_sorted = {}
    count = 0
    for k,v in anno_index_dict.items():
      anno_index_dict_sorted[count] = anno_index_dict[k]
      count += 1

    return anno_index_dict_sorted


  def create_image_anno_index(self,anno_index):
    """Create image-based annotation index"""
    image_anno_index = {}
    for class_label_index, anno_obj in anno_index.items():
      for region_obj in anno_obj['regions']:
        fn = region_obj['fn']
        # add filename if not exists
        if fn not in image_anno_index.keys():
          image_anno_index[fn] = {}
          image_anno_index[fn]['regions'] = []
          image_anno_index[fn]['filepath'] = region_obj['filepath']
          image_anno_index[fn]['basename'] = region_obj['basename']
          image_anno_index[fn]['ext'] = region_obj['ext']
        # add region/annotation
        robj = region_obj.copy()
        image_anno_index[fn]['regions'].append(robj)

    return image_anno_index





  def create_croplets(self,kwargs):
    data = json.load(kwargs['input'])
    anno_index = self.create_annotation_index(data,kwargs['image_dir'])
    self.export_croplets(anno_index,kwargs['output'])


  def append_croplets(self,anno_index,width=160):
    """Add croplet (cropped annotation) to the annotation index object
    :param  annos: serialized index of annotations
    returns:  the annos with dict key for croplet
    """
    for class_idx,anno_obj in tqdm(anno_index.items()):
      for region in anno_obj['regions']:
        im = cv.imread(region['filepath'])
        h,w = im.shape[:2]
        sx = max(0,region['x'])
        sy = max(0,region['y'])
        sw = min(1,region['width'])
        sh = min(1,region['height'])
        x1 = int(sx * w)
        y1 = int(sy * h)
        x2 = int((sw * w) + x1)
        y2 = int((sh * h) + y1)
        im = im[y1:y2,x1:x2]
        im = imutils.resize(im,width=width)
        region['croplet'] = im
    return anno_index


  def export_croplets(self,anno_index,dir_out,width=256):
    """Save croplets to disk"""
    fiox.ensure_dir(dir_out)
    anno_index_croplets = self.append_croplets(anno_index,width=width)
    for class_idx, anno_obj in anno_index_croplets.items():
      class_label_index = anno_obj['class_index']
      dir_class_idx = os.path.join(dir_out,str(class_label_index))
      fiox.ensure_dir(dir_class_idx)
      for region in anno_obj['regions']:
        try:
          im = region['croplet']
          anno_id = region['id']
          fp_croplet = os.path.join(dir_class_idx,'{}.jpg'.format(anno_id))
          cv.imwrite(fp_croplet,im)
        except:
          print('[-] Error. No croplet.')
          return False
    return True
    

        