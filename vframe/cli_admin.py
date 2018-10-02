# --------------------------------------------------------
# This is the vframe administrative script for utility
# add/edit commands in vframe/admin/commands directory
# --------------------------------------------------------
import click

from vframe.settings import vframe_cfg as cfg
from vframe.utils import logger_utils
from vframe.models.click_factory import ClickSimple

# click cli factory
cc = ClickSimple.create(cfg.DIR_COMMANDS_PROCESSOR_ADMIN)

# --------------------------------------------------------
# CLI
# --------------------------------------------------------
@click.group(cls=cc, chain=False)
@click.option('-v', '--verbose', 'verbosity', count=True, default=4, 
  show_default=True,
  help='Verbosity: -v DEBUG, -vv INFO, -vvv WARN, -vvvv ERROR, -vvvvv CRITICAL')
@click.pass_context
def cli(ctx, **kwargs):
  """\033[1m\033[94mVFRAME: Admin/Utility Scripts\033[0m                                                
  """
  ctx.opts = {}
  # init logger
  logger_utils.Logger.create(verbosity=kwargs['verbosity'])



# --------------------------------------------------------
# Entrypoint
# --------------------------------------------------------
if __name__ == '__main__':
    cli()

