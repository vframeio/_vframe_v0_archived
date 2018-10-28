"""
File utilities
"""
import sys
import os
from os.path import join
import stat

from glob import glob
from pprint import pprint
import shutil
import distutils
import pathlib
from pathlib import Path
import json
import csv
import pickle
import threading
from queue import Queue
import time
import logging
import itertools
import collections

import hashlib
import pymediainfo
import click
from tqdm import tqdm
import cv2 as cv
from PIL import Image
import imutils

from vframe.settings import vframe_cfg as cfg
from vframe.settings import types
#from vframe.models.mapping_item import SugarcubeMappingItem
from vframe.models.media_item import MediaRecordItem

log = logging.getLogger(cfg.LOGGER_NAME)


# ------------------------------------------
# File I/O read/write little helpers
# ------------------------------------------

def zpad(x, num_zeros=cfg.ZERO_PADDING):
  return str(x).zfill(num_zeros)

def get_ext(fpp, lower=True):
  """Retuns the file extension w/o dot
  :param fpp: (Pathlib.path) filepath
  :param lower: (bool) force lowercase
  :returns: (str) file extension (ie 'jpg')
  """
  fpp = ensure_posixpath(fpp)
  ext = fpp.suffix.replace('.', '')
  return ext.lower() if lower else ext


def convert(fp_in, fp_out):
  """Converts between JSON and Pickle formats
  Pickle files are about 30-40% smaller filesize
  """
  if get_ext(fp_in) == get_ext(fp_out):
    log.error('Input: {} and output: {} are the same. Use this to convert.')

  lazywrite(lazyload(fp_in), fp_out)


def load_csv(fp_in, as_list=True):
  """Loads CSV and retuns list of items
  :param fp_in: string filepath to CSV
  :returns: list of all CSV data
  """ 
  if not Path(fp_in).exists():
    log.info('loading {}'.format(fp_in))
  log.info('loading: {}'.format(fp_in))
  with open(fp_in, 'r') as fp:
    items = csv.DictReader(fp)
    if as_list:
      items = [x for x in items]
    log.info('returning {:,} items'.format(len(items)))
    return items


def lazywrite(data, fp_out, sort_keys=True):
  """Writes JSON or Pickle data"""
  ext = get_ext(fp_out)
  if ext == 'json':
    return write_json(data, fp_out, sort_keys=sort_keys)
  elif ext == 'pkl':
    return write_pickle(data, fp_out)
  else:
    raise NotImplementedError('[!] {} is not yet supported. Use .pkl or .json'.format(ext))


def lazyload(fp_in, ordered=True):
  """Loads JSON or Pickle serialized data"""
  if not Path(fp_in).exists():
    log.error('file does not exist: {}'.format(fp_in))
    return {}
  ext = get_ext(fp_in)
  if ext == 'json':
    items = load_json(fp_in)
  elif ext == 'pkl':
    items = load_pickle(fp_in)
  else:
    raise NotImplementedError('[!] {} is not yet supported. Use .pkl or .json'.format(ext))

  if ordered:
    return collections.OrderedDict(sorted(items.items(), key=lambda t: t[0]))
  else:
    return items


def load_text(fp_in):
  with open(fp_in, 'rt') as fp:
    lines = fp.read().rstrip('\n').split('\n')
  return lines

def load_json(fp_in):
  """Loads JSON and returns items
  :param fp_in: (str) filepath
  :returns: data from JSON
  """
  if not Path(fp_in).exists():
    log.error('file does not exist: {}'.format(fp_in))
    return {}
  log.debug('loading: {}'.format(fp_in))
  with open(str(fp_in), 'r') as fp:
    data = json.load(fp)
  return data


def load_pickle(fp_in):
  """Loads Pickle and returns items
  :param fp_in: (str) filepath
  :returns: data from JSON
  """
  if not Path(fp_in).exists():
    log.error('file does not exist: {}'.format(fp_in))
    return {}
  with open(str(fp_in), 'rb') as fp:
    data = pickle.load(fp)
  return data


def load_records(fp_in, slice_idxs=None, media_record_type=types.MediaRecord.SHA256, verbose=False):
  """Loads JSON or Pickle mapping data based on file extension
  :param fp_in: (str) of serialized items
  :param mapping: enum of ItemMappingType
  :returns: (dict) of mapped/non-mapped objects
  """
  # reads in serialized JSON data
  records = lazyload(fp_in, ordered=True)

  # slice data
  if slice_idxs:
    records = dict(itertools.islice(records.items(), slice_idxs[0], slice_idxs[1]))
  # remap/unserialize items to modeled data
  if media_record_type is types.MediaRecord.SHA256:
    records = {k: MediaRecordItem.from_dict(v) for k, v in tqdm(records.items())}
  else:
    msg = '{} is not valid'.format(type(media_record_type))
    log.error(msg)
    raise TypeError(msg)
  
  # important: mapping files are always sorted by SHA256 key
  records = order_items(records)

  return records

def chair_to_mapping(chair_items):
  """Converts ChairItem list to MappingItem dict"""
  return order_items({x.sha256: x.media_record for x in chair_items})

def order_items(records):
  """Orders records by ASC SHA256"""
  return collections.OrderedDict(sorted(records.items(), key=lambda t: t[0]))

def write_text(data, fp_out, ensure_path=True):
  if not data:
    log.error('no data')
    return
    
  if ensure_path:
    mkdirs(fp_out)
  with open(fp_out, 'w') as fp:
    if type(data) == list:
      fp.write('\n'.join(data))
    else:
      fp.write(data)


def write_pickle(data, fp_out, ensure_path=True):
  """
  """
  if ensure_path:
    mkdirs(fp_out) # mkdir
  with open(fp_out, 'wb') as fp:
    pickle.dump(data, fp)


def write_json(data, fp_out, minify=True, ensure_path=True, sort_keys=True):
  """
  """
  if ensure_path:
    mkdirs(fp_out)
  with open(fp_out, 'w') as fp:
    if minify:
      json.dump(data, fp, separators=(',',':'), sort_keys=sort_keys)
    else:
      json.dump(data, fp, indent=2, sort_keys=sort_keys)
  log.info('wrote json: {}'.format(fp_out))

def write_csv(data, fp_out, header=None):
  """ """
  with open(fp_out, 'w') as fp:
    writer = csv.DictWriter(fp, fieldnames=header)
    writer.writeheader()
    if type(data) is dict:
      for k, v in data.items():
        fp.writerow('{},{}'.format(k, v))


def write_serialized_items(items, fp_out, ensure_path=True, minify=True, sort_keys=True):
  """Writes serialized data
  :param items: (dict) a sha256 dict of MappingItems
  :param serialize: (bool) serialize the data
  :param ensure_path: ensure the parent directories exist
  :param minify: reduces JSON file size
  """
  log.info('Writing serialized data...')
  fpp_out = ensure_posixpath(fp_out)
  serialized_items = {k: v.serialize() for k, v in tqdm(items.items()) }
  # write data
  ext = get_ext(fpp_out)
  if ext == 'json':
    write_json(serialized_items, fp_out, ensure_path=ensure_path, minify=minify, sort_keys=sort_keys)
  elif ext == 'pkl':
    write_pickle(serialized_items, fp_out)
  else:
    raise NotImplementedError('[!] {} is not yet supported. Use .pkl or .json'.format(ext))
  log.info('Wrote {:,} items to {}'.format(len(items), fp_out))


def write_modeled_data(data, fp_out, ensure_path=False):
  """
  """
  fpp_out = ensure_posixpath(fp_out)
  if ensure_path:
    mkdirs(fpp_out)
  ext = get_ext(fpp_out)
  if ext == 'pkl':
    write_pickle(data, str(fp_out))
  else:
    raise NotImplementedError('[!] {} is not yet supported. Use .pkl or .json'.format(ext))


# ---------------------------------------------------------------------
# Filepath utilities
# ---------------------------------------------------------------------

def ensure_posixpath(fp):
  """Ensures filepath is pathlib.Path
  :param fp: a (str, LazyFile, PosixPath)
  :returns: a PosixPath filepath object
  """
  if type(fp) == str:
    fpp = Path(fp)
  elif type(fp) == click.utils.LazyFile:
    fpp = Path(fp.name)
  elif type(fp) == pathlib.PosixPath:
    fpp = fp
  else:
    raise TypeError('{} is not a valid filepath type'.format(type(fp)))
  return fpp


def mkdirs(fp):
  """Ensure parent directories exist for a filepath
  :param fp: string, Path, or click.File
  """
  fpp = ensure_posixpath(fp)
  fpp = fpp.parent if fpp.suffix else fpp
  fpp.mkdir(parents=True, exist_ok=True)


def ext_media_format(ext):
  """Converts file extension into Enum MediaType
  param ext: str of file extension"
  """
  for media_format, exts in cfg.VALID_MEDIA_EXTS.items():
    if ext in exts:
      return media_format
  raise ValueError('{} is not a valid option'.format(ext))


def sha256(fp_in, block_size=65536):
  """Generates SHA256 hash for a file
  :param fp_in: (str) filepath
  :param block_size: (int) byte size of block
  :returns: (str) hash
  """
  sha256 = hashlib.sha256()
  with open(fp_in, 'rb') as fp:
    for block in iter(lambda: f.read(block_size), b''):
      sha256.update(block)
  return sha256.hexdigest()


def sha256_tree(sha256):
  """Split hash into branches with tree-depth for faster file indexing
  :param sha256: str of a sha256 hash
  :returns: str with sha256 tree with '/' delimeter
  """
  branch_size = cfg.HASH_BRANCH_SIZE
  tree_size = cfg.HASH_TREE_DEPTH * branch_size
  sha256_tree = [sha256[i:(i+branch_size)] for i in range(0, tree_size, branch_size)]
  return '/'.join(sha256_tree)


def migrate(fmaps, threads=1, action='copy', force=False):
  """Copy/move/symlink files form src to dst directory
  :param fmaps: (dict) with 'src' and 'dst' filepaths
  :param threads: (int) number of threads
  :param action: (str) copy/move/symlink
  :param force: (bool) force overwrite existing files
  """
  log = log
  num_items = len(fmaps)
  
  def copytree(src, dst, symlinks = False, ignore = None):
    # ozxyqk: https://stackoverflow.com/questions/22588225/how-do-you-merge-two-directories-or-move-with-replace-from-the-windows-command
    if not os.path.exists(dst):
      mkdirs(dst)
      # os.makedirs(dst)
      shutil.copystat(src, dst)
    lst = os.listdir(src)
    if ignore:
      excl = ignore(src, lst)
      lst = [x for x in lst if x not in excl]
    for item in lst:
      s = os.path.join(src, item)
      d = os.path.join(dst, item)
      if symlinks and os.path.islink(s):
        if os.path.exists(d):
          os.remove(d)
        os.symlink(os.readlink(s), d)
        try:
          st = os.lstat(s)
          mode = stat.S_IMODE(st.st_mode)
          os.lchmod(d, mode)
        except:
          pass # lchmod not available
      elif os.path.isdir(s):
        copytree(s, d, symlinks, ignore)
      else:
        shutil.copy(s, d)

  assert(action in ['copy','move','symlink'])

  if threads > 1:
    # threaded
    task_queue = Queue()
    print_lock = threading.Lock()

    def migrate_action(fmap):
      data_local = threading.local()
      data_local.src, data_local.dst = (fmap['src'], fmap['dst'])
      data_local.src_path = Path(data_local.src)
      data_local.dst_path = Path(data_local.dst)
      
      if force or not data_local.dst_path.exists():
        if action == 'copy':
          shutil.copy(data_local.src, data_local.dst)
          #if data_local.src_path.is_dir():
          #  copytree(data_local.src, data_local.dst)
          #else:
        elif action == 'move':
          shutil.move(data_local.src, data_local.dst)
        elif action == 'symlink':
          if force:
            data_local.dst_path.unlink()
          Path(data_local.src).symlink_to(data_local.dst)

    def process_queue(num_items):
      # TODO: progress bar
      while True:
        fmap = task_queue.get()
        migrate_action(fmap)
        log.info('migrate: {:.2f} {:,}/{:,}'.format( 
          (task_queue.qsize() / num_items)*100, task_queue.qsize(), num_items))
        task_queue.task_done()

    # avoid race conditions by creating dir structure here
    log.info('create directory structure')
    for fmap in tqdm(fmaps):
      mkdirs(fmap['dst'])

    # init threads
    for i in range(threads):
      t = threading.Thread(target=process_queue, args=(num_items,))
      t.daemon = True
      t.start()

    # process threads
    start = time.time()
    for fmap in fmaps:
      task_queue.put(fmap)

    task_queue.join()

  else:
    # non-threaded
    for fmap in tqdm(fmaps):
      mkdirs(fmap['dst'])
      if action == 'copy':
        shutil.copy(fmap['src'], fmap['dst'])
      elif action == 'move':
        shutil.move(fmap['src'], fmap['dst'])
      elif action == 'symlink':
        if force:
          Path(fmap['dst'].unlink())
        Path(fp_src).symlink_to(fp_dst)
  return










# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#
# DEPRECATED
#
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

def route_file_mappings(fmaps, dir_src, dir_dst):
  """Convert file mappings to filepaths
  :param fmaps
  """
  files = []
  for f in tqdm(fmaps):
    sha256 = f['sha256']
    ext = f['ext']
    sha256_tree = fiox.sha256_tree(sha256)
    fp_src = join(dir_src, sha256_tree, '{}.{}'.format(sha256, ext))
    fp_dst = join(dir_dst, sha256_tree, '{}.{}'.format(sha256, ext))
    files.append( {'src': fp_src, 'dst': fp_dst} )
  return files


def route_metadata_extractor(fmaps, dir_src, dir_dst):
  """Convert file mappings to filepaths
  """
  files = []
  for f in fmaps:
    sha256 = f['sha256']
    sha256_tree = fiox.sha256_tree(sha256)
    fp_src = join(dir_src, sha256_tree, '{}.{}'.format(sha256,f['ext']))
    fp_dst = join(dir_dst, sha256_tree, sha256, 'index.json')
    files.append( {'src':fp_src, 'dst':fp_dst} )
  return files


def route_metadata(fmaps, dir_src):
  """Convert file mappings to filepaths
  """
  files = []
  for f in fmaps:
    sha256 = f['sha256']
    sha256_tree = fiox.sha256_tree(sha256)
    fp_src = join(dir_src, sha256_tree, sha256, 'index.json')
    files.append( fp_src )
  return files


def route_keyframes(fmaps, dir_keyframes, dir_images, frame_size='lg'):
  """Converts file mappings to filepaths for keyframe index.json files
  :param dir_index: path to keyframe index files
  :param dir_images: path to keyframe image files
  :returns: dict with filepath and sha256
  """
  file_routes = {}
  skipped_files = []
  for f in tqdm(fmaps):
    sha256 = f['sha256']
    fp_keyframe = join(dir_keyframes, fiox.sha256_tree(sha256), sha256, 'index.json')
    # read keyframe JSON metadata
    with open(fp_keyframe,'r') as fp:
      all_indices = json.load(fp)
    try:
      basic_indices = all_indices['basic']
    except:
      skipped_files.append(fp_keyframe)
    
    # iterate frame numbers, appending to frame dict
    file_routes[sha256] = {}
    for frame_num in basic_indices:
      frame_num = zpad(frame_num)
      sha256_tree = fiox.sha256_tree(sha256)
      file_routes[sha256] = join(dir_images, sha256_tree, sha256, 
        frame_num, frame_size, 'index.jpg')

  return file_routes


def route_keyframe_image(files, dir_src):
  """Converts list of 
  :param fmaps: list of keyframe
  :param dir_src: path to keyframe image files
  """
  files = []
  for f in fmaps:
    sha256 = f['sha256']
    sha256_tree = fiox.sha256_tree(sha256)
    fp_src = join(dir_src, sha256_tree, sha256, 'index.json')
    files.append( fp_src )
  return files    


def route_file_paths(fpaths,dir_src,dir_dst):
  """Convert list of files to src:dst maps, replacing src with dst filepath
  :param fpaths: list of globbed filepaths
  :param dir_src: filepath (str)
  :param dir_dst: filepath (str)
  :return: list of dicts with src:dst filepaths (str)
  """
  return [ {'src': f, 'dst': f.replace(dir_src,dir_dst)} for f in fpaths]

def filter_routes(fmaps, action, force=False):
  """
  Create file routes as dictionary with 'src' and 'dst'
  """
  assert(action in ['write','copy','move','symlink'])
  if action == 'move' and force:
    print('[!] Warning: can not force a "move" action"')

  print('Filtering file routes for {} files...'.format(len(fmaps)))
  routes = []
  # create file maps
  for f in tqdm(fmaps):
    fp_src = f['src']
    fp_dst = f['dst']
    fpp_src = Path(fp_src)
    fpp_dst = Path(fp_dst)
    # check if exists
    valid = (action == 'write' and (force or not(fpp_dst.exists()))) or \
            (action == 'copy' and (force or not(fpp_dst.exists()))) or \
            (action == 'move' and not(fpp_dst.exists())) or \
            (action == 'symlink') and (force or not(fpp_dst.is_symlink()))
    if valid:
      routes.append( {'src': fp_src, 'dst': fp_dst} )

  return routes

