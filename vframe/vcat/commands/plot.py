"""
Creates plot of loss during YOLO training
uses code from https://github.com/Jumabek/darknet_scripts/

"""
from pathlib import Path
import click
import datetime

import matplotlib.pyplot as plt
import matplotlib.animation as anim

from vframe.utils import click_utils
from vframe.settings import types

# --------------------------------------------------------
# Plot training output
# --------------------------------------------------------
@click.command('plot', short_help='Plots training logfile')
@click.option('-i', '--input', 'fp_in',
  required=True, 
  help='Path to input logfile')
@click.option('--xmin', 'opt_xmin', default=0, type=float)
@click.option('--xmax', 'opt_xmax', default=50000, type=float)
@click.option('--ymin', 'opt_ymin', default=0, type=float)
@click.option('--ymax', 'opt_ymax', default=100, type=float)
@click.option('--animate/--static', 'opt_animate',
  is_flag=True, default=True,
  help='Animate plot')
@click.option('--delay', 'opt_interval',
  default=60,
  help='Interval for plot refresh in seconds')
@click.option('--autofit', 'opt_autofit', is_flag=True, default=False,
  help='Autofit x,y axis')
@click.pass_context
def cli(ctx, fp_in, opt_xmin, opt_xmax, opt_ymin, opt_ymax, 
  opt_animate, opt_interval, opt_autofit):
  """Add mappings data to chain"""

  # can stop training when it reaches ~0.067

  # ----------------------------------------------------------
  # imports
  

  from vframe.utils import logger_utils, im_utils, file_utils

  log = logger_utils.Logger.getLogger()
  

  # ----------------------------------------------------------
  # proces

  
  iters = []
  loss = []
  
  plt.style.use('ggplot')
  fig, ax = plt.subplots(figsize=(12,6))
           

  def update(i):
    """Updates animation"""

    numbers = {'1','2','3','4','5','6','7','8','9'}
    log.debug('updating')

    with open(fp_in, 'r') as f:
      lines  = [line.rstrip("\n") for line in f.readlines()]
    log.info('read in {:,}'.format(len(lines)))
    
    iters = []
    loss = []

    # from https://github.com/Jumabek/darknet_scripts/
    for line in lines:
      try:
        args = line.split(' ')
        if args[0][-1:] == ':' and args[0][0] in numbers :
          a = int(args[0][:-1])
          b = float(args[2])
          iters.append(a)
          loss.append(b)
      except Exception as ex:
        log.debug('error parsing line: {}'.format(line))
        log.error('{}'.format(ex))

    ax.clear()

    log.debug('iters: {}'.format(len(iters)))
    title = Path(fp_in).stem
    plt.title('YoloV3: {}'.format(title))
    now = datetime.datetime.now()
    suptitle = now.strftime("%b %d %Y %H:%M:%S")
    plt.suptitle(suptitle)
    plt.xlabel('Iters')
    plt.ylabel('Loss')
    plt.grid()

    if not opt_autofit:
      ax.set_xlim([opt_xmin, opt_xmax])
      ax.set_ylim([opt_ymin, opt_ymax])
    ax.plot(iters,loss)


  if opt_animate:
    log.info('animating')
    # unsure what frames option does
    a = anim.FuncAnimation(fig, update, frames=1, repeat=True, interval=opt_interval * 1000)
  else:
    log.info('not animating')
    update(0)

  plt.show()
