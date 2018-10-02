import os
from os.path import join
from pathlib import Path
import logging
import collections
import threading
import time
import threading
from queue import Queue

import click
from tqdm import tqdm
import cv2 as cv
from PIL import Image

from vframe.settings import vframe_cfg as cfg
from vframe.utils import im_utils, file_utils
from vframe.models.metadata_item import KeyframeMetadataType
from vframe.models.enums import MetadataType

def extract(items, dir_out, dir_videos, keyframe_type, threads=1):
  """Extracts keyframes from images"""
  
  task_queue = Queue()
  print_lock = threading.Lock()
  log = logging.getLogger()

  if threads > 1:
    
    def thread_processor(task_obj):
      tl = threading.local()
      tl.fp_video = task_obj['fp_video']
      tl.idxs = task_obj['idxs']
      tl.dir_out = task_obj['dir_out']
      tl.sha256_tree = task_obj['sha256_tree']
      tl.sha256 = task_obj['sha256']
      try:
        tl.frame_ims = im_utils.vid2frames(tl.fp_video, idxs=tl.idxs)
      except Exception as ex:
        logging.getLogger().error('Could not read video file')
        logging.getLogger().error('file: {}'.format(tl.fp_video))
        logging.getLogger().error('sha256: {}'.format(tl.sha256))
        return
        
      tl.labels = cfg.IMAGE_SIZE_LABELS
      tl.sizes = cfg.IMAGE_SIZES

      for tl.k_label, tl.k_width in zip(reversed(tl.labels), reversed(tl.sizes)):
        tl.label = tl.labels[tl.k_label]
        tl.width = tl.sizes[tl.k_width]
        # pyramid down frame sizes 1280, 640, 320, 160
        try:
          tl.frame_ims = [im_utils.resize(tl.im, width=tl.width) for tl.im in tl.frame_ims]
        except:
          logging.getLogger().error('')
          logging.getLogger().error('Could not resize. Bad video or missing file')
          logging.getLogger().error(tl.sha256)
          logging.getLogger().error('')
          return


        for tl.idx, tl.im in zip(tl.idxs, tl.frame_ims):
          # ensure path exists
          tl.zpad = file_utils.zpad(tl.idx)
          tl.fp_dst = join(tl.dir_out, tl.sha256_tree, tl.sha256, tl.zpad, tl.label, 'index.jpg')
          # convert to PIL
          tl.im_pil = im_utils.ensure_pil(tl.im, bgr2rgb=True)
          file_utils.ensure_path_exists(tl.fp_dst)
          tl.im_pil.save(tl.fp_dst, quality=cfg.JPG_SAVE_QUALITY)


    def process_queue(num_items):
      # TODO: progress bar
      while True:
        task_obj = task_queue.get()
        thread_processor(task_obj)
        logging.getLogger().info('process: {:.2f}% {:,}/{:,}'.format( 
          (task_queue.qsize() / num_items)*100, num_items-task_queue.qsize(), num_items))
        task_queue.task_done()

    # avoid race conditions by creating dir structure here
    log.info('create directory structure first to avoid race conditions')
    log.info('TODO: this needs to be fixed, thread lock maybe')
    for sha256, item in tqdm(items.items()):
      item_metadata = item.metadata.get(MetadataType.KEYFRAME, {})
      sha256_tree = file_utils.sha256_tree(sha256)
      fp_dst = join(dir_out, sha256_tree)
      file_utils.ensure_path_exists(fp_dst)

    # init threads
    num_items = len(items)
    for i in range(threads):
      t = threading.Thread(target=process_queue, args=(num_items,))
      t.daemon = True
      t.start()

    # process threads
    start = time.time()
    for sha256, item in items.items():
      sha256_tree = file_utils.sha256_tree(sha256)
      item_metadata = item.metadata.get(MetadataType.KEYFRAME, {})
      if not item_metadata:
        continue
      keyframe_data = item_metadata.metadata
      idxs = keyframe_data.get(keyframe_type)
      fp_video = join(dir_videos, sha256_tree, '{}.{}'.format(sha256, item.ext))
      task_obj = {
        'fp_video': fp_video,
        'idxs': idxs,
        'dir_out':dir_out,
        'sha256': sha256,
        'sha256_tree': sha256_tree
        }
      task_queue.put(task_obj)

    task_queue.join()

  else:
    
    for sha256, item in tqdm(items.items()):
      item_metadata = item.metadata.get(MetadataType.KEYFRAME, {})
      if not item_metadata:
        continue

      sha256_tree = file_utils.sha256_tree(sha256)
      keyframe_data = item_metadata.metadata
      
      #idxs_basic = keyframe_data.get(KeyframeMetadataType.BASIC)
      #idxs_dense = keyframe_data.get(KeyframeMetadataType.DENSE)
      #idxs_expanded = keyframe_data.get(KeyframeMetadataType.EXPANDED)

      # fetches the metadata by the enum type from the custom click param
      idxs = keyframe_data.get(keyframe_type)

      # get frames from video
      fp_video = join(dir_videos, sha256_tree, '{}.{}'.format(sha256, item.ext))
      frame_ims = im_utils.vid2frames(fp_video, idxs=idxs)
      labels = cfg.IMAGE_SIZE_LABELS
      sizes = cfg.IMAGE_SIZES
      for k_label, k_width in zip(reversed(labels), reversed(sizes)):
        label = labels[k_label]
        width = sizes[k_width]
        # pyramid down frame sizes 1280, 640, 320, 160
        frame_ims = [im_utils.resize(im, width=width) for im in frame_ims]

        for idx, im in zip(idxs, frame_ims):
          # ensure path exists
          zpad = file_utils.zpad(idx)
          fp_dst = join(dir_out, sha256_tree, sha256, zpad, label, 'index.jpg')
          # conver to PIL
          im_pil = im_utils.ensure_pil(im, bgr2rgb=True)
          file_utils.ensure_path_exists(fp_dst)
          im_pil.save(fp_dst, quality=cfg.JPG_SAVE_QUALITY)
    