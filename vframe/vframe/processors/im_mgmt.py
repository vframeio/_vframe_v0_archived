import os
import sys
from os.path import join
from pathlib import Path
import csv
import shutil
import json
from tqdm import tqdm
from glob import glob
from pathlib import Path
import cv2 as cv
from PIL import Image
import imutils
# local
sys.path.append('/vframe/tools')
from config import settings as cfg
from . import fiox
from . import imx


class ImageManagement:

  def __init__(self):
    pass

  def create_video_web_images(self,kwargs):
    """create resized JPGs for web"""

    files = json.load(kwargs['mappings'])
    print('[+] Processing {} files'.format(len(files)))

    names = cfg.WEB_IMAGE_NAMES
    sizes = cfg.WEB_IMAGE_SIZES

    # resize image
    for f in tqdm(files):
      # construct source filepath
      sha256 = f['sha256']
      ext = f['ext']
      sha256_tree = fiox.sha256_tree(sha256)
      dir_src = join(kwargs['json'],sha256_tree,sha256)
      fp_json = join(dir_src,'index.json')
      if not Path(fp_json).exists():
        print('[-] Error. No JSON for {}'.format(fp_json))
        continue

      with open(fp_json,'r') as fp:
        keyframe_scenes = json.load(fp)

      # load the video
      fp_video = join(kwargs['videos'],sha256_tree,'{}.{}'.format(sha256,ext))
      cap = cv.VideoCapture(fp_video)

      # for each keyframe in kwargs['scene'] (default,dense,expanded)
      try:
        keyframes = keyframe_scenes[kwargs['scene']]
      except:
        print('[-] Error. No keyframes for {}'.format(fp_video))
        continue
      for idx in keyframes:
        cap.set(cv.CAP_PROP_POS_FRAMES, idx-1)
        res, im = cap.read()
        # make all sizes
        idx_name = str(idx).zfill(cfg.ZERO_PADDING)
        for abbr,w in zip(names,sizes):
          fp_dir_out = join(kwargs['output'],sha256_tree,sha256,idx_name,abbr)
          fiox.ensure_dir(fp_dir_out)
          fp_im = join(fp_dir_out,'index.jpg')
          im_pil = imx.ensure_pil(im,bgr2rgb=True)
          w_orig,h_orig = im_pil.size
          h = int((w / w_orig) * h_orig)
          im_pil = im_pil.resize((w,h), Image.ANTIALIAS)
          #im_pil.save(fp_im, 'PNG', quality=)
          im_pil.save(fp_im, 'JPEG', quality=kwargs['quality'])


  def create_photos_web_images(self,kwargs):
    """create resized JPGs for web"""

    files = json.load(kwargs['mappings'])
    print('[+] Processing {} files'.format(len(files)))

    names = cfg.WEB_IMAGE_NAMES
    sizes = cfg.WEB_IMAGE_SIZES

    # resize image
    for f in tqdm(files[:1]):
      # construct source filepath
      sha256 = f['sha256']
      ext = f['ext']
      sha256_tree = fiox.sha256_tree(sha256)
      fp_src = join(kwargs['input'],sha256_tree,'{}{}'.format(sha256,ext))
      try:
        im = cv.imread(fp_src)
      except:
        print('[-] Could not load: {}'.format(fp_src))
        continue
      if im is None or im.shape[0] == 0:
        print('[-] Bad file: {}'.format(fp_src))
        continue

      # make all sizes
      for abbr,w in zip(names,sizes):
        fp_dir_out = join(kwargs['output'],sha256_tree,sha256,abbr)
        fiox.ensure_dir(fp_dir_out)
        fp_im = join(fp_dir_out,'index.jpg')
        im_pil = imx.ensure_pil(im,bgr2rgb=True)
        w_orig,h_orig = im_pil.size
        h = int((w / w_orig) * h_orig)
        im_pil = im_pil.resize((w,h), Image.ANTIALIAS)
        #im_pil.save(fp_im, 'PNG', quality=)
        im_pil.save(fp_im, 'JPEG', quality=kwargs['quality'])
