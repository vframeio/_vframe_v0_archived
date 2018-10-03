import sys

import click
import cv2 as cv

from vframe.utils import logger_utils

log = logger_utils.Logger.getLogger()

def _pause(ctx):
  ctx.opts['paused'] = True
  log.info("paused - hit space to resume, or h for help")
  # click.echo( "paused - hit space to resume, or h for help", err=True)


def handle_keyboard(ctx, opt_delay):
  """Handle keyboard input using opencv handlers"""

  ctx.opts.setdefault('display_previous', False)
  ctx.opts.setdefault('paused', False)

  while True:

    k = cv.waitKey(opt_delay) & 0xFF
    
    if k == 27 or k == ord('q'):  # ESC
      # exits the app
      cv.destroyAllWindows()
      sys.exit('Exiting because Q or ESC was pressed')
    elif k == ord(' '):
      if ctx.opts['paused']:
        ctx.opts['paused'] = False
        break
      else:
        _pause(ctx)
    if k == 81: # left arrow key
      log.info('previvous not yet working')
      break
    elif k == 83: # right arrow key
      log.info('next')
      break
    elif k == ord('h'):
        print ("""
      keyboard controls:
          q           : quit
          h           : show this help
          space       : pause/unpause
      """)
    if not ctx.opts['paused']:
      break