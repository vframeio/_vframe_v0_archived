"""

"""
import click

from vframe.utils import click_utils
from vframe.settings import types
from vframe.utils import logger_utils, im_utils, file_utils


# --------------------------------------------------------
# testing
# --------------------------------------------------------
@click.command()
@click.option('-i', '--input', 'dir_project', required=True,
  help='Path to project directory')
@click.option('--cropped/--full', 'opt_cropped', is_flag=True, default=True,
  help='Save cropped or full image')
@click.pass_context
def cli(ctx, dir_project, opt_cropped):
  """Generates croplets of annotations for review"""

  # ------------------------------------------------
  # imports
  from glob import glob
  from os.path import join
  from pathlib import Path

  import cv2 as cv
  from tqdm import tqdm


  log = logger_utils.Logger.getLogger()
  log.debug('generate croplets')
  
  # get list of files
  fp_labels = glob(join(dir_project, 'labels', '*.txt'))
  fp_ims = [x.replace('txt', 'jpg').replace('labels', 'images') for x in fp_labels]

  if len(fp_labels) != len(fp_ims):
    log.error('{} labels but {} images. not equal. exiting'.format(len(fp_labels), len(fp_ims)))

  fp_classes = join(dir_project, 'classes.txt')
  class_names = file_utils.load_text(fp_classes)
  slug_lookup = {x: i for i, x in enumerate(class_names)}

  dir_croplets = join(dir_project, 'croplets')
  for class_name in class_names:
    file_utils.mkdirs(join(dir_croplets, class_name))

  for fp_im, fp_label in zip(tqdm(fp_ims), fp_labels):
    count = 0
    annos = file_utils.load_text(fp_label)

    if not annos or len(annos) < 1 or annos == ['']:
      log.debug('skipping empty file')
      continue

    fn = Path(fp_im).stem
    # load image
    im = cv.imread(fp_im)
    try:
      ih, iw = im.shape[:2]
    except Exception as ex:
      log.error('could not read: {}'.format(fp_im))

    for anno in annos:
      vals = anno.split(' ')
      class_idx = int(vals[0])
      ncx, ncy, nw, nh = list(map(float, vals[1:]))
      fp_croplet = join(dir_croplets, class_names[int(class_idx)], '{}_{}.jpg'.format(fn, count))
      count += 1
      #log.debug('class_id: {}, cx: {}, cy: {}, nw: {}, nh: {}'.format(class_idx, ncx, ncy, nw, nh))
      # TODO change to BBox
      cx = iw * ncx
      cy = ih * ncy
      w = iw * nw
      h = ih * nh
      x1 = int(cx - w / 2)
      y1 = int(cy - h / 2)
      x2 = int(x1 + w)
      y2 = int(y1 + h)
      if opt_cropped:
        im_croplet = im[y1:y2, x1:x2]
      else:
        # draw box on full image
        im_croplet = cv.rectangle(im.copy(), (x1, y1), (x2, y2) , (0, 255, 0), thickness=2)
      cv.imwrite(fp_croplet, im_croplet)
      break
