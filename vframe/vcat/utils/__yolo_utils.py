"""
Utility scripts to generate YOLO training data
"""
import os
from os.path import join
import sys
import json
import numpy as np
import random
from PIL import Image
from pprint import pprint
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import itertools
import shutil
from pathlib import Path
from utils import fiox
from training.vcat.vcat_utils import VcatUtils


class YoloUtils:

  def __init__(self):
    pass

  def generate_image_list(self,anno_objs,dir_project_images):
    # for each object-class
    ims = []
    for class_idx, anno_obj in anno_objs.items():
      for regions in anno_obj['regions']:
        ims.append('{}/{}{}'.format(dir_project_images,regions['fn'],regions['ext']))
    return ims


  def generate_yolo_config(self,yolo_version,n_classes,cfg_type,n_subdiv=16,n_batch=64):

    # load existing cfg file


    # (Generally filters depends on the classes, num and coords
    # i.e. equal to (classes + coords + 1)*num, where num is number of anchors)
    # for yolov2 (num_classes + (num_coords +1)) * num_anchors

    # load the template from github.com/pjreddie/darknet
    cwd = os.path.dirname(os.path.realpath(__file__))
    fp_cfg_orig = os.path.join(cwd,'cfgs','yolov{}.cfg'.format(yolo_version)) 
    
    if yolo_version == '2':
    
      num_filters = (n_classes + 5) * 5
      s_replace_base = [
        ('filters=425', 'filters={}'.format(num_filters)),
        ('classes=80', 'classes={}'.format(n_classes)),
        ]

      if cfg_type == 'train':
        s_replace = s_replace_base.copy()
        s_replace.append(('batch=1','#batch=1'))
        s_replace.append(('subdivisions=1\n','#subdivisions=1\n'))
        s_replace.append(('# batch=64','batch={}'.format(n_batch)))
        s_replace.append(('# subdivisions=8','subdivisions={}'.format(n_subdiv)))
      elif cfg_type == 'test':
        # change nclasses, but keep the subdiv and batch size at default
        s_replace = s_replace_base.copy()
        
    elif yolo_version == '3':
      # load the template from github.com/pjreddie/darknet
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
    
    # search and replace, return list of lines
    cfg_orig = open(fp_cfg_orig,'r')    
    cfg_new_data = []
    for line in cfg_orig:
      for query,targ in s_replace:
        if query in line:
          line = line.replace(query,targ)
      cfg_new_data.append(line.replace('\n',''))

    return cfg_new_data



  def create_label_anno_index(self,image_anno_index):
    """Create index of labels with Darknet label format
      One text file per image with one line for each annotation
    """
    label_anno_index = {}
    
    for fn_key,image_obj in image_anno_index.items():
      bname = os.path.splitext(image_obj['basename'])[0]
      fname = os.path.splitext(bname)[0]
      im_anno_txt = []
      for region_obj in image_obj['regions']:
        w = float(region_obj['width'])
        h = float(region_obj['height'])
        cx = (w/2.) + float(region_obj['x'])
        cy = (h/2.) + float(region_obj['y'])
        class_label_index = region_obj['class_label_index']
        txt = '{} {} {} {} {}'.format(class_label_index,cx,cy,w,h)
        im_anno_txt.append(txt)
        if region_obj['is_attribute']:
          # get parent id-->class_label_index
          parent_class_label_index = region_obj['parent_class_label_index']
          txt = '{} {} {} {} {}'.format(parent_class_label_index,cx,cy,w,h)
          im_anno_txt.append(txt)

      label_anno_index[fn_key] = im_anno_txt

    return label_anno_index


  def create_image_symlinks(self,image_anno_index,dir_project_images):
    """Create symlinks to image files so project can be 
    contained in single directory"""
    for fn_key,image_obj in image_anno_index.items():
        src = image_obj['filepath'] # path to local file
        dst = os.path.join(dir_project_images,'{}{}'.format(fn_key,image_obj['ext']))
        # remove symlink if exists
        if os.path.isfile(dst):
            os.remove(dst)
        # add symlink
        os.symlink(src,dst)

  def get_random_image(self,annos):
    """Return a random image for testing"""
    # get random class image
    annos_with_regions = [a for k,a in annos.items() if len(a['regions']) >1]
    v = random.choice(annos_with_regions)
    region_obj = random.choice(v['regions'])
    return region_obj['filepath']


  def create_project(self,kwargs):
    
    vcat_utils = VcatUtils()

    # load vcat JSON
    vcat_data = json.load(kwargs['input'])

    # hierarchy is 
    hierarchy = vcat_data['hierarchy']

    # if exclusions, remove classes
    if kwargs['exclude'] is not None:
        exclusions = [int(i) for i in kwargs['exclude'].split(',')]
        print('[+] Excluding: {}'.format(exclusions))
        # remove from hierarchy
        hierarchy_tmp = hierarchy.copy()
        for obj_class_id, obj_class_meta in hierarchy_tmp.items():
          if int(obj_class_id) in exclusions:
            print('[+] Removing ID: {} ({})'.format(obj_class_id,obj_class_meta['slug']))
            del hierarchy[obj_class_id]

    # class label index lookup (YOLO  ID --> VCAT ID)
    class_label_index_lkup = vcat_utils.create_class_label_index(vcat_data,kwargs)



    print('----Classes----')
    for k,v in class_label_index_lkup.items():
      print('VCAT: {} --> YOLO: {}, Label: {}'.format(k,v,hierarchy[str(k)]['slug']))

    # create object class lookup
    anno_index = vcat_utils.create_annotation_index(
      vcat_data, class_label_index_lkup, kwargs)

    for a in anno_index:
      print('Annotation index: {}'.format(a))

    sys.exit()

    # inflate annotation index to include parent class (ghost) annotations
    # TODO

    # create image index lookup
    image_anno_index = vcat_utils.create_image_anno_index(anno_index)

    # Create randomize, class-based splits for train and valid
    anno_splits = vcat_utils.create_splits(anno_index,split_val=kwargs['split_val'])
    annos_train = anno_splits['train']
    annos_valid = anno_splits['valid']
    
    # Statistics
    n_regions_train = np.sum([len(v['regions']) for k,v in annos_train.items()])
    n_regions_test = np.sum([len(v['regions']) for k,v in annos_valid.items()])
    print('[+] Regions train: {}, Test: {}'.format(n_regions_train, n_regions_test))

    print('----Classes----')
    for k,v in class_label_index_lkup.items():
      print('VCAT: {} --> YOLO: {}, Label: {}'.format(k,v,hierarchy[str(k)]['slug']))

    print('----Train----')
    for k,v in annos_train.items():
      print('{}\t{}'.format(len(v['regions']),v['slug_vcat']))

    print('----Validate----')
    for k,v in annos_valid.items():
      print('{}\t{}'.format(len(v['regions']),v['slug_vcat']))

    print('----Labels----')
    for k,v in class_label_index_lkup.items():
      print('{}\t{}'.format(k,hierarchy[str(k)]['slug']))

    sys.exit()
    
    # ---------------------------------------------------    
    # Filenames and paths
    # ---------------------------------------------------

    # convenience vars
    n_classes = len(anno_index)
    yolov = str(kwargs['yolo_version'])
    project_name = kwargs['name']
    dir_output = kwargs['dir_output']
    dir_project = os.path.join(dir_output,project_name)
    
    # create project_name directory
    fiox.ensure_dir(dir_project)

    # create images directory
    dir_project_images = os.path.join(dir_project,'images')
    fiox.ensure_dir(dir_project_images)
        
    # Create dirs if not exist
    dir_labels = os.path.join(dir_project,'labels')
    fiox.ensure_dir(dir_labels)

    # create dir for weights/backup
    dir_weights = os.path.join(dir_project,'weights')
    fiox.ensure_dir(dir_weights)


    # ---------------------------------------------------    
    # Create "classes.txt"
    # ---------------------------------------------------

    fp_classes = os.path.join(dir_project,'classes.txt')
    class_labels = []
    for k,v in class_label_index_lkup.items():
      # maps vcat id to yolo id
      slug = hierarchy[str(k)]['slug']
      slug_sp = slug.split(':')
      if len(slug_sp) > 1:
        display_name = '{} ({})'.format(slug_sp[0],slug_sp[1])
      else:
        display_name = slug_sp[0]

      class_labels.append(display_name)

    with open(fp_classes,'w') as fp:
      fp.write('\n'.join(class_labels))


    # ---------------------------------------------------    
    # Create "train.txt", "valid.txt", "classes.txt"
    # ---------------------------------------------------

    fp_train = os.path.join(dir_project,'train.txt')
    im_list = self.generate_image_list(annos_train,dir_project_images)
    with open(fp_train,'w') as fp:
      fp.write('\n'.join(im_list))

    fp_valid = os.path.join(dir_project,'valid.txt')
    im_list = self.generate_image_list(annos_valid,dir_project_images)
    with open(fp_valid,'w') as fp:
      fp.write('\n'.join(im_list))
    

    # ---------------------------------------------------    
    # Crate sym-linked image files
    # ---------------------------------------------------

    # TODO remove folder if exists
    
    self.create_image_symlinks(image_anno_index,dir_project_images)


    # ---------------------------------------------------    
    # Create "labels.txt"
    # ---------------------------------------------------

    # labels.txt contains one line per annotation
    # "1 0.434 0.605 0.2980 0.37222"
    # class_idx, cx, cy, w, h
    # for YoloV3 add parent class hierarchy to the label

    label_anno_index = self.create_label_anno_index(image_anno_index)

    for fname, anno_obj in label_anno_index.items():
      fp_txt = os.path.join(dir_labels,'{}.txt'.format(fname))
      with open(fp_txt,'w') as fp:
        fp.write('\n'.join(anno_obj))


    # ---------------------------------------------------    
    # Create "meta.data"
    # ---------------------------------------------------

    txts = []
    txts.append('classes = {}'.format(n_classes))
    txts.append('train = {}'.format(fp_train))
    txts.append('valid = {}'.format(fp_valid))
    txts.append('names = {}'.format(fp_classes))
    txts.append('backup = {}'.format(dir_weights))
    
    fp_meta = os.path.join(dir_project,'meta.data')
    with open(fp_meta,'w') as fp:
      fp.write('\n'.join(txts))


    # ---------------------------------------------------    
    # create train and test .cfg files
    # ---------------------------------------------------
    
    cfg_train_data = self.generate_yolo_config(yolov,n_classes,'train',
      n_subdiv=kwargs['subdivisions'],n_batch=kwargs['batch_size'])
    cfg_test_data = self.generate_yolo_config(yolov,n_classes,'test')

    # write data to new .cfg files
    fp_cfg_train = os.path.join(dir_project,'yolov{}.cfg'.format(yolov))
    fp_cfg_test = os.path.join(dir_project,'yolov{}_test.cfg'.format(yolov))

    with open(fp_cfg_train, 'w') as fp:
      fp.write('\n'.join(cfg_train_data))
    
    with open(fp_cfg_test, 'w') as fp:
      fp.write('\n'.join(cfg_test_data))


    # ---------------------------------------------------    
    # Create shell scripts to start and resume training
    # ---------------------------------------------------

    txts_base = []
    txts_base.append('#!/bin/bash')
    txts_base.append('DARKNET=/opt/darknet_pjreddie/darknet')
    txts_base.append('DIR_PROJECT={}'.format(dir_project))
    txts_base.append('FP_CFG={}'.format(fp_cfg_train))
    txts_base.append('FP_META={}'.format(fp_meta))
    txts_base.append('CMD="detector train"')
    txts_base.append('LOGFILE="2>&1 | tee train_log.txt"')
    
    # run_train_init.sh
    fp_sh_train_init = os.path.join(dir_project,'run_train_init.sh')
    
    txts_init = txts_base.copy()
    txts_init.append('GPUS="-gpus {}"'.format(kwargs['gpu_init']))
    txts_init.append('FP_WEIGHTS={}'.format(kwargs['init_weights']))
    txts_init.append('$DARKNET $CMD $FP_META $FP_CFG $FP_WEIGHTS $GPUS')

    with open(fp_sh_train_init,'w') as fp:
      fp.write('\n'.join(txts_init))
    

    # run_train_resume.sh
    fp_sh_train_resume = os.path.join(dir_project,'run_train_resume.sh')
    
    txts_resume = txts_base.copy()
    txts_resume.append('GPUS="-gpus {}"'.format(kwargs['gpu_full']))
    txts_resume.append('FP_WEIGHTS={}/{}.backup'.format(dir_weights,project_name))
    txts_resume.append('$DARKNET $CMD $FP_META $FP_CFG $FP_WEIGHTS $GPUS')
    
    with open(fp_sh_train_resume, 'w') as fp:
      fp.write('\n'.join(txts_resume))


    # run_test.sh
    fp_sh_test_image = os.path.join(dir_project,'run_test_image.sh')
    
    txts = []
    txts.append('#!/bin/bash')
    txts.append('DIR_DARKNET=/opt/darknet_pjreddie/')
    txts.append('DIR_PROJECT={}'.format(dir_project))
    txts.append('FP_CFG={}'.format(fp_cfg_test))
    txts.append('FP_META={}'.format(fp_meta))
    rn_filepath = self.get_random_image(annos_valid)
    txts.append('FP_IMG={}'.format(rn_filepath))
    txts.append('FP_WEIGHTS={}/{}_900.weights'.format(dir_weights,project_name))
    txts.append('#FP_WEIGHTS={}/{}_10000.weights'.format(dir_weights,project_name))
    txts.append('#FP_WEIGHTS={}/{}_30000.weights'.format(dir_weights,project_name))
    txts.append('CMD="detector test"')
    txts.append('cd $DIR_DARKNET')
    txts.append('./darknet $CMD $FP_META $FP_CFG $FP_WEIGHTS $FP_IMG')
    
    with open(fp_sh_test_image, 'w') as fp:
      fp.write('\n'.join(txts))    

    print('[+] Wrote Darknet files for {}'.format(project_name))
    print('[+] Project path: {}'.format(dir_project))


  # ------------------------------------------
  # Plot YOLO Loss Data
  # ------------------------------------------


  def create_plot(self,kwargs):
    """Create plot of Yolo Loss Avg"""
    # can stop training when it reaches ~0.067
    # TODO make dynamic plot

    f = kwargs['input']
    lines  = [line.rstrip("\n") for line in f.readlines()]
    
    numbers = {'1','2','3','4','5','6','7','8','9'}

    iters = []
    loss = []
    
    fig,ax = plt.subplots()

    for line in lines:
      args = line.split(' ')
      if args[0][-1:]==':' and args[0][0] in numbers :
        iters.append(int(args[0][:-1]))            
        loss.append(float(args[2]))
             
    ax.plot(iters,loss)
    ax.set_xlim([kwargs['xmin'],kwargs['xmax']])
    ax.set_ylim([kwargs['ymin'],kwargs['ymax']])
    project_name = os.path.basename(os.path.dirname(kwargs['input'].name))
    plt.title('YoloV3: {}'.format(project_name))
    plt.xlabel('Iters')
    plt.ylabel('Loss')
    plt.grid()
    
    plt.show()
