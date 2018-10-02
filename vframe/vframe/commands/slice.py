import click

from cli_vframe import processor


# --------------------------------------------------------
# Slice the item chain
# --------------------------------------------------------
@click.command('slice', short_help='Slice items')
@click.argument('opt_start', default=None, type=int)
@click.argument('opt_end', default=None, type=int)
@processor
@click.pass_context
def cli(ctx, sink, opt_start, opt_end):
  """Slices the item list"""

  from tqdm import tqdm
  from vframe.utils import logger_utils
    
  log = logger_utils.Logger.getLogger()
  # accumulate items
  items = []
  while True:
    try:
      items.append( (yield) )
    except GeneratorExit as ex:
      break
  
  # simple array slice on items list
  log.info('slice from {} to {}'.format(opt_start, opt_end))
  if opt_start is not None and opt_end is not None:
    items = items[opt_start:opt_end]
  else:
    log.error('error: no slice start/end')

  log.info('sliced items: {:,}'.format(len(items)))

  # update items
  ctx.opts['num_items'] = len(items)

  # rebuild the generator
  for item in tqdm(items):
    sink.send(item)
