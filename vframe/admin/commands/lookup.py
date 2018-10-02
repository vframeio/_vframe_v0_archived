"""
Looks up info about Sugarcube item
"""
import click

from vframe.settings import types
from vframe.utils import click_utils
from vframe.settings import vframe_cfg as cfg


@click.command()
@click.option('-i', '--input', 'fp_in', default=None,
  help='Override file input path')
@click.option('-m', '--metadata', 'opt_metadata_type',
  default=None, # default uses original mappings
  type=cfg.MetadataVar,
  help=click_utils.show_help(types.Metadata))
@click.option('--type', 'opt_type', required=True,
  type=cfg.SearchParamVar,
  default=click_utils.get_default(types.SearchParam.SHA256),
  help=click_utils.show_help(types.SearchParam))
@click.option('--verified', 'opt_verified',
  type=cfg.VerifiedVar,
  default=click_utils.get_default(types.Verified.VERIFIED),
  show_default=True,
  help=click_utils.show_help(types.Verified))
@click.option('--id', 'opt_id', required=True,
  help='ID value to search for')
@click.option('-d', '--disk', 'opt_disk',
  default=click_utils.get_default(types.DataStore.SSD),
  type=cfg.DataStoreVar,
  show_default=True,
  help=click_utils.show_help(types.DataStore))
@click.option('-e', '--ext', 'opt_format',
  default=click_utils.get_default(types.FileExt.PKL),
  type=cfg.FileExtVar,
  show_default=True,
  help=click_utils.show_help(types.FileExt))
@click.pass_context
def cli(ctx, fp_in, opt_metadata_type, opt_type, opt_verified, opt_id, opt_disk, opt_format):
  """search for info with ID"""

  # -------------------------------------------------
  # imports

  import os
  from os.path import join
  from pathlib import Path

  from vframe.utils import file_utils, logger_utils
  from vframe.settings.paths import Paths
  from vframe.settings import vframe_cfg as cfg

  # -------------------------------------------------
  # process 

  log = logger_utils.Logger.getLogger()
  log.debug('opt_type: {}, opt_metadata_type: {}'.format(opt_type, opt_metadata_type))

  # if not opt_type:
  #   # auto guess
  #   nchars = len(opt_id)
  #   if nchars == 64:
  #     opt_types = [types.SearchParam.SA_ID, types.SearchParam.SHA256]
  #   elif nchars == 32:
  #     opt_type = [types.SearchParam.MD5]
  #   else:
  #     log.error('id not a valid format. use either 32-hex MD5 or 64-hex SHA256') 
  #     return
    
  if not fp_in:
    if opt_metadata_type:
      fp_in = Paths.metadata_index(data_store=opt_disk, file_format=opt_format, 
        verified=opt_verified, metadata_type=opt_metadata_type)
      # use source media_records
    else:
      fp_in = Paths.media_record_index(data_store=opt_disk, file_format=opt_format, 
        verified=opt_verified)


  log.info('opening: {}'.format(fp_in))

  media_records = file_utils.load_records(fp_in)

  log.info('searching {:,} media records for {}: {}'.format(len(media_records), opt_type, opt_id))

  found_items = []
  for sha256, media_record in media_records.items():

    if opt_type == types.SearchParam.SHA256:
      # quick match sha256
      if opt_id == sha256:
        found_items.append(media_record)
        break
    else:
      # get sc metadata
      sugarcube_metadata = media_record.get_metadata(types.Metadata.SUGARCUBE)
      
      if not sugarcube_metadata:
        log.error('no sugarcube metadata. Try "append -t sugarcube"')
        return

      # match other params
      if opt_type == types.SearchParam.SA_ID:
        if opt_id == sugarcube_metadata.sa_id:
          found_items.append(media_record)
          break 
      elif opt_type == types.SearchParam.MD5:
        if opt_id == sugarcube_metadata.md5:
          found_items.append(media_record)
          break 

  if not len(found_items) > 0:
    log.error('No results')
  else:
    log.info('{} item found'.format(len(found_items)))

    metadata_records = media_record.metadata
    log.debug('sha256: {}'.format(media_record.sha256))
    log.debug('\tformat: {}'.format(media_record.media_format))
    log.debug('\tverified: {}'.format(media_record.verified))
    if opt_metadata_type:
      for metadata_type, metadata_obj in metadata_records.items():
        log.debug('\ttype: {}'.format(metadata_type))
        try:
          log.debug('\tmetadata: {}'.format(metadata_obj.serialize()))
        except Exception as ex:
          log.debug('\tmetadata: {}'.format(metadata_obj.__dict__))
