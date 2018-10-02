"""
Generates the initial media records
"""

import click

from vframe.settings import types
from vframe.utils import click_utils
from vframe.settings import vframe_cfg as cfg

from cli_vframe import generator

@click.command('gen_records', short_help='Generates MediaRecords')
@click.option('-i', '--input', 'fp_in', required=True, type=str,
    help="Path to input date for record system (eg sugarcube CSV)")
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
  default=click_utils.get_default(types.DataStore.SSD),
  type=cfg.DataStoreVar,
  show_default=True,
  help=click_utils.show_help(types.DataStore))
@click.option('--media-format', 'opt_media_format_type',
  default=click_utils.get_default(types.MediaFormat.VIDEO),
  type=cfg.MediaFormatVar,
  show_default=True,
  help=click_utils.show_help(types.MediaFormat))
@generator
@click.pass_context
def cli(ctx, sink, fp_in, opt_media_record_type, opt_client_record_type, opt_disk, opt_media_format_type):
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
  log = logger_utils.Logger.getLogger()
  
  log.debug('fp_in: {}'.format(fp_in))
  log.debug('opt_disk: {}'.format(opt_disk))
  log.debug('opt_media_format_type: {}'.format(opt_media_format_type))
  log.debug('opt_media_record_type: {}'.format(opt_media_record_type))
  log.debug('opt_verified: {}'.format(ctx.opts['verified']))
  
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

    opt_verified = ctx.opts.get('verified')
    # sa_id,sha256,md5,location,verified
    csv_rows = file_utils.load_csv(fp_in) # as list

    # remap as sugarcube item
    records = {}
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
          records[sha256] = MediaRecordItem(sha256, media_format, verified)
    
    log.debug('non-filtered: {:,} records'.format(len(records)))

    # -------------------------------------------------

    for sha256, records in records.items():
      sink.send(ChairItem(ctx, records))









# filter by verified/unverified
# if not opt_all:
#   opt_verified = ctx.opts['verified']
#   log.debug('filtering to keep only: {}'.format(opt_verified))
#   records = {k: v for k, v in records.items() if v.verified == opt_verified}
#   log.debug('filtered: {:,} records'.format(len(records)))

# # filter by media type
# if opt_media_type is not None:
#   log.debug('filtering to keep only: {}'.format(opt_media_type))
#   records = {k: v for k, v in records.items() if v.media_type == opt_media_type}
#   log.debug('filtered: {:,} records'.format(len(records)))
