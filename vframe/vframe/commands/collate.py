"""
Collates singe .json files to unified record (deprecated)
"""
import click

from vframe.settings import types
from vframe.utils import click_utils
from vframe.settings import vframe_cfg as cfg

from cli_vframe import processor

# --------------------------------------------------------
# Collate deprecated metadata-tree JSON files
# --------------------------------------------------------
@click.command('collate', short_help='Collate metadata-tree items')
@click.option('-d', '--disk', 'opt_disk',
  default=click_utils.get_default(types.DataStore.HDD),
  type=cfg.DataStoreVar,
  show_default=True,
  help=click_utils.show_help(types.DataStore))
@click.option('-e', '--ext', 'opt_format',
  default=click_utils.get_default(types.FileExt.PKL),
  type=cfg.FileExtVar,
  show_default=True,
  help=click_utils.show_help(types.FileExt))
@click.option('-t', '--type', 'opt_metadata_tree_type',
  default=None, # default uses original mappings
  type=cfg.MetadataTreeVar,
  show_default=True,
  required=True,
  help=click_utils.show_help(types.MetadataTree))
@processor
@click.pass_context
def cli(ctx, sink, opt_disk, opt_format, opt_metadata_tree_type):
  """Collate depated metadata tree files"""

  # -------------------------------------------------
  # imports
  
  import click
  from pathlib import Path
  from tqdm import tqdm

  from vframe.settings import vframe_cfg as cfg
  from vframe.settings.paths import Paths
  from vframe.utils import file_utils, logger_utils
  from vframe.models.metadata_item import MediainfoMetadataItem, KeyframeMetadataItem

  from cli_vframe import processor

  
  # -------------------------------------------------
  # process

  log = logger_utils.Logger.getLogger()

  if opt_metadata_tree_type == types.MetadataTree.MEDIAINFO_TREE:
    metdata_type = types.Metadata.MEDIAINFO
  if opt_metadata_tree_type == types.MetadataTree.KEYFRAME_TREE:
    metdata_type = types.Metadata.KEYFRAME
  
  dir_metadata = Paths.metadata_tree_dir(opt_metadata_tree_type, data_store=opt_disk)

  # accumulate chair items
  chair_items = []
  while True:
    try:
      chair_items.append( (yield) )
    except GeneratorExit as ex:
      break

  skipped = []
  num_skipped = 0
  found = []
  num_found = 0

  # iterate chair items and gather metadata index.json files
  num_items = len(chair_items)
  for chair_item in tqdm(chair_items):
    item = chair_item.item
    sha256 = item.sha256
    sha256_tree = file_utils.sha256_tree(sha256)
    fpp_metadata = Path(dir_metadata, sha256_tree, sha256, 'index.json')
    
    # skip if not existing 
    metadata = {}
    if fpp_metadata.exists():
      try:
        metadata = file_utils.lazyload(fpp_metadata)
      except Exception as ex:
        log.error('could not read json: {}, ex: {}'.format(str(fpp_metadata), ex))
        continue
    
    # count items skipped
    if not metadata:
      skipped.append(fpp_metadata)
      num_skipped = len(skipped)
      per = num_skipped / (num_found + num_skipped) * 100
      # log.debug('{:.2f}% ({:,}/{:,}) not found: {}'.format(per, num_skipped, (num_found + num_skipped), str(fpp_metadata)))
      log.debug('{:.2f}% ({:,}/{:,}) missing'.format(per, num_skipped, (num_found + num_skipped)))
      chair_item.item.set_metadata(metdata_type, metadata_obj)
    else:
      found.append(fpp_metadata)
      num_found = len(found)
      # construct and append metadata
      if metdata_type == types.Metadata.MEDIAINFO:
        metadata_obj = MediainfoMetadataItem.from_index_json(metadata)
        chair_item.item.set_metadata(metdata_type, metadata_obj)
      elif metdata_type == types.Metadata.KEYFRAME:
        metadata_obj = KeyframeMetadataItem.from_index_json(metadata)
        chair_item.item.set_metadata(metdata_type, metadata_obj)
      else:
        raise ValueError('{} is not a valid metadata type'.format(metdata_type))

  log.info('skipped: {:,} items'.format(len(skipped)))


  # -------------------------------------------------
  # rebuild

  for chair_item in chair_items:
      sink.send(chair_item)
