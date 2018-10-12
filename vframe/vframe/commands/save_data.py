"""
Saves metadata
"""
import click

from vframe.settings import vframe_cfg as cfg
from vframe.settings import types
from vframe.utils import click_utils

from cli_vframe import processor

chair_items = []
# --------------------------------------------------------
# Save metadata to JSON/Pickle
# --------------------------------------------------------
@click.command('save', short_help='Write items to JSON|Pickle')
@click.option('-o', '--output', 'fp_out', default=None,
  help='Path to output file (overrides other settings)')
@click.option('-e', '--ext', 'opt_format',
  default=click_utils.get_default(types.FileExt.PKL),
  type=cfg.FileExtVar,
  show_default=True,
  help=click_utils.show_help(types.FileExt))
@click.option('-d', '--disk', 'opt_disk',
  default=click_utils.get_default(types.DataStore.SSD),
  type=cfg.DataStoreVar,
  show_default=True,
  help=click_utils.show_help(types.DataStore))
@click.option('-t', '--type', 'opt_metadata_type',
  type=cfg.MetadataVar,
  required=False,  # unless output is not provided
  show_default=True,
  help=click_utils.show_help(types.Metadata))
@click.option('--minify/--no-minify', 'opt_minify', default=False,
  help='Minify output if using JSON')
@click.option('-f', '--force', 'opt_force', is_flag=True,
  help='Force overwrite')
@click.option('--suffix', 'opt_suffix', default=None,
  help='Force overwrite')
@click.option('--checkpoints', 'opt_ckpt_size', type=int, default=None,
  help='Checkpoint save interval')
@click.option('--purge/--no-purge', 'opt_purge', is_flag=True, default=True,
  help='Purge metadata after copying data to save queue')
@processor
@click.pass_context
def cli(ctx, sink, fp_out, opt_format, opt_disk, opt_metadata_type, 
  opt_minify, opt_force, opt_suffix, opt_ckpt_size, opt_purge):
  """Writes items to disk as JSON or Pickle"""

  
  # ------------------------------------------------------
  # imports

  import sys
  from os.path import join
  from pathlib import Path
  from collections import OrderedDict
  import gc
  import copy
  import numpy as np

  from vframe.settings.paths import Paths
  from vframe.utils import file_utils, click_utils
  from vframe.utils import logger_utils
  from vframe.models.chair_item import MediaRecordChairItem
  

  # --------------------------------------------------------
  # init

  log = logger_utils.Logger.getLogger()

  if not fp_out:
    fp_out = Paths.metadata_index(opt_metadata_type, data_store=opt_disk, 
      file_format=opt_format, verified=ctx.opts['verified'])
  
  fpp_out = Path(fp_out)
  
  if opt_suffix:
    fp_out = join(str(fpp_out.parent), '{}_{}{}'.format(fpp_out.stem, opt_suffix, fpp_out.suffix))
  
  def create_ckpt_fpaths(num_items, opt_ckpt_size):
    ckpts = list(range(0, num_items, opt_ckpt_size))
    if np.max(np.array(ckpts)) < num_items:
      ckpts.append(num_items)

    for i, ckpt in enumerate(ckpts[:-1]):
      n_start = file_utils.zpad(ckpt, num_zeros=cfg.CKPT_ZERO_PADDING)
      n_end = file_utils.zpad(ckpts[i+1], num_zeros=cfg.CKPT_ZERO_PADDING)
      ckpt_suffix = 'ckpt_{}_{}{}'.format(n_start, n_end, fpp_out.suffix)  # 0_10.pkl
      fp = join(str(fpp_out.parent), '{}_{}'.format(fpp_out.stem, ckpt_suffix))
      ckpt_fpaths.append(fp)

    return ckpt_fpaths

  # --------------------------------------------------------
  # checkpoint interval saving
  
  if opt_ckpt_size:
   
    # save items every N iterations
    yield_count = 0
    ckpt_iter_num = 0
    # chair_items = OrderedDict({})
    chair_items = []
    ckpt_fpaths = []

    while True:
      
      chair_item = yield
      yield_count += 1

      # ctx variables can only be accessed after processor starts
      # hack: set filepaths after while/yield loop starts
      if not ckpt_fpaths:
        num_items = ctx.opts['num_items']
        ckpt_fpaths = create_ckpt_fpaths(num_items, opt_ckpt_size)
        log.debug('{}'.format(ckpt_fpaths))
        # ensure it does not already exist
        for fp in ckpt_fpaths:
          if Path(fp).exists() and not opt_force:
            log.error('File "{}" exists. Use "-f" to override'.format(fp))
            log.error('This error occurs later because it uses variables from the processor context')
            return

      # accumulate chair items
      chair_items.append(chair_item)

      if (yield_count > 0 and yield_count % opt_ckpt_size == 0) or yield_count >= num_items:
        
        fp_out = ckpt_fpaths[ckpt_iter_num]
        # convert chair items to media records
        log.debug('chair_items: {}'.format(len(chair_items)))
        mapping_items = file_utils.chair_to_mapping(chair_items)
        # write to disk
        log.debug('fp_out: {}'.format(fp_out))
        file_utils.write_serialized_items(mapping_items, fp_out, 
          ensure_path=True, minify=opt_minify)
        
        # TODO improve this
        #
        # purge metadata,        
        for chair_item in chair_items:
          chair_item.purge_metadata()
        
        chair_items = []
        mapping_items = []
        ckpt_iter_num += 1
        # collect/empty garbage
        gc.collect()

      # continue chair processors
      sink.send(chair_item)


  else:

    # --------------------------------------------------------
    # save all 
  

    # save all items
    # exit if file exists
    if Path(fp_out).exists() and not opt_force:
      m = 'File "{}" exists. Use "-f" to override'.format(fp_out)
      log.error(m)
      return

    # accumulate items
    chair_items = []
    while True:
      try:
        chair_items.append( (yield) )
      except GeneratorExit as ex:
        break

    if not len(chair_items) > 0:
      log.error('no items to save')
      return

    # convert chair items to media records
    log.debug('chair_items: {}'.format(len(chair_items)))
    mapping_items = file_utils.chair_to_mapping(chair_items)
    log.debug('mapping_items: {}'.format(len(mapping_items)))
    
    # write to disk
    log.debug('fp_out: {}'.format(fp_out))
    file_utils.write_serialized_items(mapping_items, fp_out, ensure_path=True, minify=opt_minify)

    # rebuild the generator
    for chair_item in chair_items:
      sink.send(chair_item)