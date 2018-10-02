"""Sugarcube utility scripts
"""

import sys
import os
from os.path import join
from pathlib import Path
import json
import csv
import pickle
import logging 

import click
from tqdm import tqdm

from vframe.utils import file_utils, im_utils
from vframe.settings import vframe_cfg as cfg
from vframe.models.mapping_item import SugarcubeMappingItem


def map_items(fp_in, fp_out, verified_type=None, media_type=None, minify=True):
  """Maps and writes records from Sugarcube's CSV format into MappingItems as JSON"""

  logger = logging.getLogger()
  # load CSV
  csv_rows = file_utils.load_csv(fp_in) # as list
  # remap as sugarcube item
  logger.info('mapping {:,} items to {}'.format(len(csv_rows), SugarcubeMappingItem))
  items = {}
  # map sugarcube items
  for csv_row in tqdm(csv_rows):
    sha256 = csv_row.get('sha256','')
    if csv_row.get('location', False) and len(sha256) == 64:
      items[sha256] = SugarcubeMappingItem.from_csv_row(csv_row)
  logger.info('non-filtered: {:,} items'.format(len(items)))
  # filter by verified/unverified
  if verified_type is not None:
    logger.info('filtering to keep only: {}'.format(verified_type))
    items = {k: v for k, v in items.items() if v.verified == verified_type}
    logger.info('filtered: {:,} items'.format(len(items)))
  # filter by media type
  if media_type is not None:
    logger.info('filtering to keep only: {}'.format(media_type))
    items = {k: v for k, v in items.items() if v.media_type == media_type}
    logger.info('filtered: {:,} items'.format(len(items)))

  # write items to JSON or Pickle
  file_utils.write_serialized_items(items, fp_out, ensure_path=True, minify=minify)
   

def lookup(fp_in, hash_id, hash_type):
  """Returns serialized info about MappingItem
  :param fp_in: (str) filepath to serialized MappingItems in JSON
  :param hash_type: which attribute to search for
  :returns: (dict) with match or empty
  """
  if hash_type == 'sha256':
    return lookup_hash256(fp_in, hash_id)
  elif hash_type == 'sugarcube':
    return lookup_sugarcube_id(fp_in, hash_id)
  else:
    logging.getLogger().warn('hash type: {} is invalid'.format(hash_type))
    return {}


def lookup_hash256(fp_in, sha256, find_line=True):
  """Looks up sha256 hash and returns any info about it
  :param fp_in: (str) filepath to mappings
  :param sha256: (str) ID
  :returns: (dict) info about object
  """
  log = logging.getLogger()
  items = file_utils.load_mappings(fp_in)
  if find_line:
    for i, (k, item) in tqdm(enumerate(items.items())):
      if item.sha256 == sha256:
        log.info('Found match at: {}'.format(i))  
        return item
  else:
    if sha256 in items.keys():
      return SugarcubeMappingItem.from_dict(items[sha256]).serialize()  
    else:
      return {}


def lookup_sugarcube_id(fp_in, hash_id):
  """Looks up Sugarcube ID and returns any info about it
  :param fp_in: (str) filepath to mappings
  :param sugarcube_id: (str) ID
  :returns: (dict) info about object
  """
  log = logging.getLogger()
  items = file_utils.load_mappings(fp_in)
  for i, (sha256, item) in tqdm(enumerate(items.items())):
    if item['sugarcube_id'] == hash_id:
      log.info('Found match at: {}'.format(i))
      return SugarcubeMappingItem.from_dict(item).serialize()  
  return {}
