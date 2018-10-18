"""
Generates the initial media records
"""

import click

from vframe.settings import types
from vframe.utils import click_utils
from vframe.settings import vframe_cfg as cfg

@click.command('gen_records', short_help='Generates MediaRecords')
@click.option('-i', '--input', 'fp_in', required=True, type=str,
    help="Path to input date for record system (eg sugarcube CSV)")
@click.option('-o', '--output', 'fp_out', default=None,
  help='Path to output file (overrides other settings)')
@click.option('--record-type', 'opt_media_record_type',
  type=cfg.MediaRecordVar,
  default=click_utils.get_default(types.MediaRecord.SHA256),
  show_default=True,
  help=click_utils.show_help(types.MediaRecord))
@click.option('--client-type', 'opt_client_record_type',
  type=cfg.ClientRecordVar,
  default=click_utils.get_default(types.ClientRecord.SUGARCUBE),
  show_default=True,
  help=click_utils.show_help(types.ClientRecord))
@click.option('-d', '--disk', 'opt_disk',
  default=click_utils.get_default(types.DataStore.HDD),
  type=cfg.DataStoreVar,
  show_default=True,
  help=click_utils.show_help(types.DataStore))
@click.option('--media-format', 'opt_media_format_type',
  default=click_utils.get_default(types.MediaFormat.VIDEO),
  type=cfg.MediaFormatVar,
  show_default=True,
  help=click_utils.show_help(types.MediaFormat))
@click.option('-e', '--ext', 'opt_format',
  default=click_utils.get_default(types.FileExt.PKL),
  type=cfg.FileExtVar,
  show_default=True,
  help=click_utils.show_help(types.FileExt))
@click.option('--status', 'opt_verified',
  type=cfg.VerifiedVar,
  show_default=True,
  default=click_utils.get_default(types.Verified.VERIFIED),
  help=click_utils.show_help(types.Verified))
@click.option('--minify/--no-minify', 'opt_minify', default=False,
  help='Minify output if using JSON')
@click.option('-f', '--force', 'opt_force', is_flag=True,
  help='Force overwrite')
@click.pass_context
def cli(ctx, fp_in, fp_out, opt_media_record_type, opt_client_record_type, opt_disk, 
  opt_media_format_type, opt_format, opt_verified, opt_minify, opt_force):
  """Generates dataset records"""
  
  # greeet

  # 
  import os
  from os.path import join
  from pathlib import Path

  from tqdm import tqdm
  
  from vframe.settings.paths import Paths
  from vframe.utils import file_utils, logger_utils
  from vframe.settings import vframe_cfg as cfg
  from vframe.models.media_item import MediaRecordItem
  from vframe.models.chair_item import ChairItem


  # -------------------------------------------------
  # process here
  metadata_type = types.Metadata.MEDIA_RECORD
  log = logger_utils.Logger.getLogger()
  if not fp_out:
    fp_out = Paths.metadata_index(metadata_type, data_store=opt_disk, 
      file_format=opt_format, verified=opt_verified)

  log.debug('fp_in: {}'.format(fp_in))
  log.debug('fp_in: {}'.format(fp_out))
  log.debug('opt_disk: {}'.format(opt_disk))
  log.debug('opt_media_format_type: {}'.format(opt_media_format_type))
  log.debug('opt_media_record_type: {}'.format(opt_media_record_type))
  log.debug('opt_verified: {}'.format(opt_verified))
  
  # input error handling
  if opt_media_format_type == types.MediaFormat.PHOTO:
    log.error('Option not available: {}'.format(types.MediaFormat.PHOTO))
    return
  if opt_media_record_type != types.MediaRecord.SHA256:
    log.error('Option not available: {}'.format(opt_media_record_type))
    return
  if opt_client_record_type != types.ClientRecord.SUGARCUBE:
    log.error('Option not available: {}'.format(opt_media_record_type))
    return

  # handle different types of input records
  if opt_client_record_type == types.ClientRecord.SUGARCUBE:
    # generate records from Sugarcube client export data

    verified_status = True if opt_verified is types.Verified.VERIFIED else False
    # sa_id,sha256,md5,location,verified
    csv_rows = file_utils.load_csv(fp_in) # as list

    # remap as sugarcube item
    media_records = {}
    # map sugarcube items

    log.debug('mapping {:,} entries to {}'.format(len(csv_rows), opt_media_record_type))
    for row in tqdm(csv_rows):
      
      sha256 = row.get('sha256', None)
      fp_media = row.get('location', None)
      is_verified = row.get('verified', '').lower() == 'true'
      verified = types.Verified.VERIFIED if is_verified else types.Verified.UNVERIFIED

      if sha256 and fp_media and len(sha256) == 64 and verified == opt_verified:
        ext = file_utils.get_ext(fp_media)
        media_format = file_utils.ext_media_format(ext)  # enums.MediaType
        if media_format == opt_media_format_type:
          media_records[sha256] = MediaRecordItem(sha256, media_format, verified)
    
    log.debug('non-filtered: {:,} records'.format(len(media_records)))

    
    log.debug('fp_out: {}'.format(fp_out))
    file_utils.write_serialized_items(media_records, fp_out, 
      ensure_path=True, minify=opt_minify)

    # -------------------------------------------------