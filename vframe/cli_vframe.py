# --------------------------------------------------------
# This is the vframe chain for metadata
# add/edit commands in commands directory
# --------------------------------------------------------
from functools import update_wrapper, wraps
import logging
from pathlib import Path

import click

from vframe.settings import vframe_cfg as cfg
from vframe.settings import types
from vframe.utils import logger_utils, click_utils
from vframe.models.click_factory import ClickComplex


# click cli factory
cc = ClickComplex.create(cfg.DIR_COMMANDS_PROCESSOR_CHAIR)


# --------------------------------------------------------
# Chain CLI
# --------------------------------------------------------
@click.group(chain=True, cls=cc)
@click.option('-v', '--verbose', 'verbosity', count=True, default=4, 
  show_default=True,
  help='Verbosity: -v DEBUG, -vv INFO, -vvv WARN, -vvvv ERROR, -vvvvv CRITICAL')
@click.option('--verified/--unverified', 'opt_verified', is_flag=True, default=True,
  help='Verified or unverified media')
@click.pass_context
def cli(ctx, **kwargs):
  """\033[1m\033[94mVFRAME: Visual Forensics and Metadata Extraction\033[0m                                                
  """
  ctx.opts = {}
  # add opt_verified in beginning to allow access from tailing commands
  if kwargs['opt_verified']:
    ctx.opts['verified'] = types.Verified.VERIFIED
  else:
    ctx.opts['verified'] = types.Verified.UNVERIFIED

  # init logger
  logger_utils.Logger.create(verbosity=kwargs['verbosity'])
  

@cli.resultcallback()
def process_commands(processors, **kwargs):
    """This result callback is invoked with an iterable of all the chained
    subcommands.  As in this example each subcommand returns a function
    we can chain them together to feed one into the other, similar to how
    a pipe on unix works.

    This function was copied from click's documentation.
    """

    def sink():
      """This is the end of the pipeline"""
      while True:
        yield

    sink = sink()
    sink.__next__()
    # sink.next()
    #try:
    #  sink.__next__()
    #except Exception as ex:
    #  logging.getLogger('vframe').error('{} (can\'t view here)'.format(ex))
    #  return

    # Compose all of the coroutines, and prime each one
    for processor in reversed(processors):
      sink = processor(sink)
      sink.__next__()
      # sink.next()
      # try:
      #   sink.__next__()
      # except Exception as ex:
      #   logging.getLogger('vframe').error('error: {} (can\'t view here)'.format(ex))
      #   return

    sink.close() # this executes the whole pipeline.
                 # however it is unnecessary, as close() would be automatically
                 # called when sink goes out of scope here.



def processor(f):
    """Helper decorator to rewrite a function so that it returns another
    function from it.

    This function was copied from click's documentation.
    """
    def processor_wrapper(*args, **kwargs):
        @wraps(f)
        def processor(sink):
            return f(sink, *args, **kwargs)
        return processor
    return update_wrapper(processor_wrapper, f)


def chair_command(name=None):
    """
    This decorates chair commands.
    """
    def decorator(f):
        f = click.pass_context(f)
        def coro_wrapper(*args, **kwargs):
            def coroutine(sink):
                return f(sink, *args, **kwargs)
            return coroutine
        return chair_cli.command(name)( update_wrapper(coro_wrapper, f) )
    return decorator

def generator(f):
    """Similar to the :func:`processor` but passes through old values
    unchanged.
    """
    @processor
    def _generator(sink, *args, **kwargs):
        try:
            while True:
                sink.send((yield))
        except GeneratorExit:
            f(sink, *args, **kwargs)
            sink.close()
    return update_wrapper(_generator, f)





# --------------------------------------------------------
# Entrypoint
# --------------------------------------------------------
if __name__ == '__main__':
    cli()

