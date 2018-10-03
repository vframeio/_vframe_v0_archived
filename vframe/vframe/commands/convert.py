"""
DEPRECATED: use admin cli
Converts between Pickle and JSON files
"""
import click

from vframe.settings import types
from vframe.utils import click_utils
from vframe.settings import vframe_cfg as cfg

from cli_vframe import processor

# --------------------------------------------------------
# Converts Pickle <--> JSON
# --------------------------------------------------------
@click.command('convert_format', short_help='Converts between JSON and Pickle')
@click.option('-i', '--input', 'fp_in', type=str,
    help="Path to serialized JSON")
@click.option('-o', '--output', 'fp_out', type=str,
    help="Path to serialized Pickle")
@click.option('-d', '--disk', 'opt_disk',
  default=click_utils.get_default(types.DataStore.SSD),
  type=cfg.DataStoreVar,
  show_default=True,
  help=click_utils.show_help(types.DataStore))
@click.option('--from', 'opt_format_in',
  default=click_utils.get_default(types.FileExt.PKL),
  type=cfg.FileExtVar,
  show_default=True,
  help=click_utils.show_help(types.FileExt))
@click.option('--to', 'opt_format_out',
  default=click_utils.get_default(types.FileExt.PKL),
  type=cfg.FileExtVar,
  show_default=True,
  help=click_utils.show_help(types.FileExt))
@click.option('-t', '--type', 'opt_metadata_type',
  default=None, # default uses original mappings
  type=cfg.MetadataVar,
  show_default=True,
  help=click_utils.show_help(types.Metadata))
@click.option('--minify/--no-minify', 'opt_minify', default=False,
  help='Minify output if using JSON')
@click.option('-f', '--force', 'opt_force', is_flag=True,
  help='Force overwrite')
@processor
@click.pass_context
def cli(ctx, sink, fp_in, fp_out, opt_disk, opt_format_in, opt_format_out, 
  opt_metadata_type, opt_minify, opt_force):
  """Converts JSON to Pickle"""
  
  # -------------------------------------------------
  # imports 

  import os
  from os.path import join
  from pathlib import Path

  import click_spinner

  from vframe.settings import vframe_cfg as cfg
  from vframe.settings.paths import Paths
  from vframe.settings import types
  from vframe.utils import file_utils, logger_utils
  

  # -------------------------------------------------
  # process

  log = logger_utils.Logger.getLogger()

  if not opt_metadata_type and not fp_in:
    # TODO create custom exception
    log.error('Error: missing option for either "-t" / "--type" or "-i" / "--input"')
    return

  if not fp_in:
    fp_in = Paths.metadata_index(opt_metadata_type, data_store=opt_disk, 
      file_format=opt_format_in, verified=ctx.opts['verified'])
      
  if not fp_out:
    fpp_in = Path(fp_in)
    ext = opt_format_out.name.lower()
    fp_out = join(str(fpp_in.parent), '{}.{}'.format(fpp_in.stem, ext))

  # check again 
  ext_in, ext_out = (file_utils.get_ext(fp_in), file_utils.get_ext(fp_out))
  if ext_in == ext_out or opt_format_in == opt_format_out:
    ctx.fail('Cannot convert from "{}" to "{}" (same)'.format(ext_in, ext_in))

  if Path(fp_out).exists() and not opt_force:
    log.error('Files exists. Use "-f/--force" to overwrite. {}'.format(fp_out))
  else:
    with click_spinner.spinner():
      log.info('Converting {} to {}'.format(fp_in, fp_out))
      if ext_out == types.FileExt.PKL.name.lower():
        file_utils.write_pickle(file_utils.load_json(fp_in), fp_out)
      elif ext_out == types.FileExt.JSON.name.lower():
        file_utils.write_json(file_utils.load_pickle(fp_in), fp_out, 
          minify=opt_minify)

    # compare sizes
    size_src = os.path.getsize(fp_in) / 1000000
    size_dst = os.path.getsize(fp_out) / 1000000
    per = size_dst / size_src * 100
    txt_verb = 'increased' if size_dst > size_src else 'decreased'
    log.info('Size {} from {:.2f}MB to {:.2f}MB ({:.2f}%)'.format(
      txt_verb, size_src, size_dst, per))


  # accumulate chair items
  chair_items = []
  while True:
    try:
      chair_items.append( (yield) )
    except GeneratorExit as ex:
      break

  
  # -------------------------------------------------
  # rebuild the generator

  for chair_item in chair_items:
    sink.send(chair_item)