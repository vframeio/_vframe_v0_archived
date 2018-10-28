"""This opens images to create generator
"""
import sys
from os.path import join
from pathlib import Path

import numpy as np
import click

from vframe.utils import logger_utils, im_utils, file_utils
from vframe.utils import click_utils
from vframe.settings import types
from vframe.settings import vframe_cfg as cfg
from vcat.settings import vcat_cfg
from vcat.utils import vcat_utils, yolo_utils


# --------------------------------------------------------
# Plot training output
# --------------------------------------------------------
@click.command()
@click.option('-i', '--input', 'fp_in', 
  default=vcat_cfg.FP_VCAT_ANNOTATIONS,
  help='VCAT annotations')
@click.option('-o', '--output', 'dir_project',
  default=vcat_cfg.DIR_PROJECT_TEST,
  help="YOLO project directory")
@click.option('--images', 'dir_images',
  required=True,
  default=vcat_cfg.DIR_IMAGES,
  help='Path to images')
@click.option('--batch', 'opt_batch', 
  default=64, 
  help='Images to process per mini-batch. Higher is better. Depends on GPU RAM)')
@click.option('--subdivisions', 'opt_subdivisions', 
  default=16,
  help='Number of times to divide batch size per mini-batch')
@click.option('--size', 'opt_size', 
  type=(int, int), default=(416, 416))
@click.option('--gpu-init', 'opt_gpu_init', 
  default="0", type=str,
    help="GPU index for initial training")
@click.option('--gpu-resume', 'opt_gpus_resume', 
  default="0,1",type=str,
  help="GPU indices for full training (after 1.000 epochs)")
@click.option('--logfile', 'opt_logfile', default='training.log', type=str,
    help="Path to logfile")
@click.option('--weights', 'opt_init_weights',
    default=vcat_cfg.FP_DARKNET_INIT_WEIGHTS,
    help="Path to YOLO V3 initialization weights")
@click.option('-e', '--exclude', 'opt_excludes', 
  type=int, multiple=True,
  help='Classes to exclude')
@click.option('--parent/--no-parents', 'opt_parent_hierarchy', 
  is_flag=True, default=True,
  help='Use hierarchical parent labeling')
@click.option('-e', '--exclude', 'opt_excludes', 
  type=int, multiple=True,
  help='Classes to exclude')
@click.pass_context
def cli(ctx, fp_in, dir_project, dir_images, opt_batch, opt_subdivisions, opt_size,
  opt_gpu_init, opt_gpus_resume, opt_logfile, opt_init_weights, 
  opt_parent_hierarchy, opt_excludes):
  """Genrates project config files"""

  # ----------------------------------------------------------
  # imports


  log = logger_utils.Logger.getLogger()
  log.debug('generate YOLO V3 project files')  

  # echo options
  log.info('fp_in: {}'.format(fp_in))
  log.info('dir_project: {}'.format(dir_project))
  log.info('opt_batch: {}'.format(opt_batch))
  log.info('opt_subdivisions: {}'.format(opt_subdivisions))
  log.info('opt_size: {}'.format(opt_size))
  log.info('opt_gpu_init: {}'.format(opt_gpu_init))
  log.info('opt_gpus_resume: {}'.format(opt_gpus_resume))
  log.info('opt_logfile: {}'.format(opt_logfile))
  log.info('opt_init_weights: {}'.format(opt_init_weights))
  log.info('opt_parent_hierarchy: {}'.format(opt_parent_hierarchy))
  

  # create project directory
  file_utils.mkdirs(dir_project)

  
  # -------------------------------------------------------------------------
  # VCAT
  # -------------------------------------------------------------------------
  
  log.debug('loading: {}'.format(fp_in))
  vcat_data = vcat_utils.load_annotations(fp_in, opt_excludes)

  # get the ordered hierarchy
  log.debug('create VCAT object class hierarchy')
  hierarchy_tree = vcat_utils.hierarchy_tree(vcat_data['hierarchy'].copy())
  anno_idx_lookup = vcat_utils.hierarchy_flat(hierarchy_tree)
  slug_lookup = {v['slug']: k for k, v in anno_idx_lookup.items()}

  # if opt, append parent clases
  if opt_parent_hierarchy:
    vcat_utils.append_parents(anno_idx_lookup, vcat_data['hierarchy'])
  
  # generate the annotations
  # YOLO needs 
  #   - labels/image.txt txt file with same name
  #   - images/image.jpg
  #   - one master train.txt file with one image filepath per line
  #   - assumes using pjreddie/darknet repo

  yolo_annos = yolo_utils.gen_annos(vcat_data)
  dir_labels = join(dir_project, 'labels')
  file_utils.mkdirs(dir_labels)

  # goup annotations by image
  anno_image_groups = {}
  for yolo_anno in yolo_annos:
    fn = yolo_anno.filename
    if fn not in anno_image_groups.keys():
      anno_image_groups[fn] = []
    anno_image_groups[fn].append(yolo_anno)

  # write one text file per image with all annotations
  for fn, image_annos in anno_image_groups.items():
    lines = [x.as_line() for x in image_annos]
    txt = '\n'.join(lines)
    fp_label = join(dir_project, 'labels', '{}.txt'.format(Path(fn).stem))
    file_utils.write_text(lines, fp_label)

  # create symlinks for images
  dir_project_images = join(dir_project, 'images')
  file_utils.mkdirs(dir_project_images)
  for fn, image_annos in anno_image_groups.items():
    # TODO: these should be PNG files for training
    fp_src = join(dir_images, fn)
    fpp_src = Path(fp_src)
    fp_dst = join(dir_project_images, fn)
    fpp_dst = Path(fp_dst)
    if fpp_dst.exists() and fpp_dst.is_symlink():
      fpp_dst.unlink()
    fpp_dst.symlink_to(fpp_src)

  # create split 80/20
  # TODO create split per each class
  annos_train = []
  annos_validate = []
  test_per = 0.2
  train_per = 1 - test_per

  anno_class_groups = {}
  for yolo_anno in yolo_annos:
    class_id = yolo_anno.class_id
    if class_id not in anno_class_groups.keys():
      anno_class_groups[class_id] = []
    anno_class_groups[class_id].append(yolo_anno)  

  for class_id, yolo_annos in anno_class_groups.items():
    n_items = len(yolo_annos)
    n_train = int(train_per * n_items)
    n_test = int(test_per * n_items)
    annos_train_idxs = np.random.choice(yolo_annos, n_train, replace=False)
    for yolo_anno in annos_train_idxs:
      annos_train.append(yolo_anno)
    annos_test_idxs = np.random.choice(yolo_annos, n_test, replace=False)
    for yolo_anno in annos_test_idxs:
      annos_validate.append(yolo_anno)

  # write the train.txt file
  log.debug('write train.txt')
  fp_train_txt = join(dir_project, 'train.txt')
  txt = []
  for yolo_anno in annos_train:
    txt.append(join(dir_project_images, '{}'.format(yolo_anno.filename)))
  txt = '\n'.join(txt)
  file_utils.write_text(txt, fp_train_txt)

  log.debug('write valid.txt')
  # write the valid.txt file
  fp_valid_txt = join(dir_project, 'valid.txt')
  txt = []
  for yolo_anno in annos_validate:
    txt.append(join(dir_project_images, '{}'.format(yolo_anno.filename)))
  txt = '\n'.join(txt)
  file_utils.write_text(txt, fp_valid_txt)

  # write the classes.txt file
  log.debug('writing classes file')
  classes_txt = ['{}'.format(v['slug'].replace(':', '_')) \
    for k, v in anno_idx_lookup.items()]
  classes_txt = '\n'.join(classes_txt)
  fp_classes = join(dir_project, 'classes.txt')
  file_utils.write_text(classes_txt, fp_classes) 
  
  # write a debugging tree file
  log.debug('writing annotations to hierarchy index')
  annos_tree = vcat_utils.hierarchy_tree_display(hierarchy_tree)
  fp_hierarchy = join(dir_project, 'hierarchy.txt')
  file_utils.write_text(annos_tree, fp_hierarchy)

  # write a debugging CSV with id, label, count
  log.debug('writing annotations to flat index')
  annos_debug_txt = ['{}, {}, {}'.format(k, v['slug'], v['region_count']) \
    for k, v in anno_idx_lookup.items()]
  annos_debug_txt = '\n'.join(annos_debug_txt)
  fp_anno_debug = join(dir_project, 'annotation_summary.txt')
  file_utils.write_text(annos_debug_txt, fp_anno_debug)


  # -------------------------------------------------------------------------
  # YOLO config files
  # -------------------------------------------------------------------------
  num_classes = len(anno_idx_lookup)
  # Generate data file
  log.debug('generate meta.data')
  metadata = yolo_utils.gen_meta(dir_project, num_classes)
  fp_meta = join(dir_project, 'meta.data')
  file_utils.write_text(metadata, fp_meta) 

  # Generate cfg file
  log.debug('generate training config')
  fp_cfg_train = join(dir_project, 'yolov3.cfg')
  cfg_data = yolo_utils.gen_cfg('train', num_classes, opt_batch, opt_subdivisions, opt_size)
  file_utils.write_text(cfg_data, fp_cfg_train)
  
  # test
  log.debug('generate test config')
  fp_cfg_test = join(dir_project, 'yolov3_test.cfg')
  cfg_data = yolo_utils.gen_cfg('test', num_classes, opt_batch, opt_subdivisions, opt_size)
  file_utils.write_text(cfg_data, fp_cfg_test)
  
  # shell scripts
  # train
  log.debug('generate init training shell script')
  fp_sh_train = join(dir_project,'run_train_init.sh')
  sh_train = yolo_utils.gen_sh_train(dir_project, fp_meta, fp_cfg_train, opt_gpu_init, weights=opt_init_weights)
  file_utils.write_text(sh_train, fp_sh_train)
  # test image
  log.debug('generate test shell script')
  fp_sh_test = join(dir_project,'run_test_image.sh')
  sh_test = yolo_utils.gen_sh_test(dir_project, fp_meta, fp_cfg_test)
  file_utils.write_text(sh_test, fp_sh_test)
  # resume training
  log.debug('generate resume training shell script')
  fp_sh_resume = join(dir_project,'run_train_resume.sh')
  sh_resume = yolo_utils.gen_sh_train(dir_project, fp_meta, fp_cfg_train, opt_gpus_resume )
  file_utils.write_text(sh_resume, fp_sh_resume)

  # make backup weights directory
  log.debug('make backup folder')
  file_utils.mkdirs(join(dir_project, 'weights'))


  # TODO: Generate a README.md file summary
  log.debug('TODO: generate README')
  fp_readme = join(dir_project, 'README.md')
  yolo_utils.gen_readme()
  # done
  log.info('generated config files in: {}'.format(dir_project))  


