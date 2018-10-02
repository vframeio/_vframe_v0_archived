#!python

import os
from os.path import join
import sys
import click
import json
import csv
from tqdm import tqdm
import hashlib
from glob import glob
import cv2 as cv
import imutils
import faiss
import pickle
import numpy as np
import pandas as pd
import sqlite3
import time
import csv
# TODO change to relative
sys.path.append('/vframe/tools/')
from utils import fiox
from utils import imx
from utils.im_mgmt import ImageManagement
from config import settings as cfg


# --------------------------------------------------------
# Click CLI parser
# --------------------------------------------------------
@click.group()
def cli():
    pass



# --------------------------------------------------------
# Resize video images
# --------------------------------------------------------
@cli.command()
@click.option('--input',required=True,type=click.File('r'),
    help="Path features .pkl")
@click.option('--output',required=True,type=click.Path(exists=True),
    help="Path to output folder (.index, timing.txt)")
@click.option('--type',required=True,type=click.Option(
  ['Flat','PCA80,Flat','IVF512,Flat','IVF512,SQ4','IVF512,SQ8','PCAR8,IMI2x10,SQ8']),
  help="FAISS factory type")
@click.option('--store-db/--no-store-db',default=True,
  help='Store result in SQLITE DB')
@click.option('--store-index/--no-store-index',default=True,
  help='Store index?')
@click.option('--append/--no-append',default=True,
  help='Append to timing output file')
def train(**kwargs):
  """Train FAISS options"""
  fp_pkl = kwargs['features']
  fpp_pkl = Path(fpp_pkl)
  feat_net_name = fpp_pkl.parent # (vgg16, resnet-18, alexnet)
  #pkl_fn = './features/{}/index.pkl'.format(dataset)
  #index_fn = "./indexes/{}-{}.index".format(dataset, factory_type.replace(",", "_"))
  #db_fn = "./indexes/{}.db".format(dataset)  
  fp_index = join(kwargs['output'],'{}.index'.format(feat_net_name))
  fp_db = join(kwargs['output'],'{}.db'.format(feat_net_name))
  fp_timing = join(kwargs['output'],'{}_timing.txt'.format(feat_net_name))

  # load features
  data = pickle.loads(kwargs['input'])

  # insert features in DB
  if kwargs['store_db']:
    print("[+] saving database...")
    db = sqlite3.connect(db_fn)
    cursor = db.cursor()
    cursor.execute('''
        DROP TABLE IF EXISTS frames
    ''')
    cursor.execute('''
        CREATE TABLE frames(id INTEGER PRIMARY KEY, sha256 TEXT, frame TEXT)
    ''')
    db.commit()
    for sha256 in data['videos'].keys():
      for frame in data['videos'][sha256].keys():
        cursor.execute('''INSERT INTO frames(sha256, frame)
                          VALUES(?,?)''', (sha256, frame))
    db.commit()

  feats = np.array([ data['videos'][v][frame] for v in data['videos'].keys() for frame in data['videos'][v].keys() ])

  n, d = feats.shape

  # process FAISS
  print("[+] processing {}x {}d features...".format(n, d))
  index = faiss.index_factory(d, factory_type)

  time_start = time.time()
  index.train(feats)
  train_time = time.time() - time_start
  print("[+] train time: {}".format(train_time))

  time_start = time.time()
  # for i in range(10):
  #   feats[:, 0] += np.arange(feats) / 1000.
  index.add(feats)
  add_time = time.time() - time_start
  print("[+] add time: {}".format(add_time))

  # print(index.is_trained)
  # print(index.ntotal)

  #if opt.store_index:
  if kwargs['store_index']:
    faiss.write_index(index, fp_index)

  keys = list(data['videos'].keys())
  key = keys[0]

  frames = list(data['videos'][key].keys())
  frame = frames[0]

  # print(key, frame)

  vec = data['videos'][key][frame]

  # print('sanity check:')

  time_start = time.time()
  D, I = index.search(np.array([vec]).astype('float32'), 15)
  search_end = time.time()
  search_time = search_end - time_start
  print("search time: {}".format(search_time))

  fmode = 'a' if kwargs['append'] else 'w'
  with open("timing.txt", fmode) as fp:
    index_size = os.path.getsize(index_fn)
    print("index size: {}".format(int(index_size/(1024*1024))))
    writer = csv.writer(fp)
    if not fmode == 'a':
      writer.writerow("#dataset, #index.ntotal, #d, #factory_type, #train_time, #add_time, #search_time, #index_size")
    writer.writerow([dataset, index.ntotal, d, factory_type, train_time, add_time, search_time, index_size])

  # print("distances:")
  # print(D[0])

  # print("indexes:")
  # print(I[0])



# --------------------------------------------------------
# Entrypoint
# --------------------------------------------------------

if __name__ == '__main__':
  cli()




# parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', required=True)
# parser.add_argument('--factory_type', required=True)
# parser.add_argument('--store_db', action='store_true')
# parser.add_argument('--store_index', action='store_true')
# opt = parser.parse_args()

# dataset = opt.dataset
# factory_type = opt.factory_type

pkl_fn = './features/{}/index.pkl'.format(dataset)
index_fn = "./indexes/{}-{}.index".format(dataset, factory_type.replace(",", "_"))
db_fn = "./indexes/{}.db".format(dataset)

fh = open(pkl_fn, 'rb')
raw = fh.read()
fh.close()
data = pickle.loads(raw)

if opt.store_db:
  print("saving database...")
  db = sqlite3.connect(db_fn)
  cursor = db.cursor()
  cursor.execute('''
      DROP TABLE IF EXISTS frames
  ''')
  cursor.execute('''
      CREATE TABLE frames(id INTEGER PRIMARY KEY, hash TEXT, frame TEXT)
  ''')
  db.commit()
  for hash in data['videos'].keys():
    for frame in data['videos'][hash].keys():
      cursor.execute('''INSERT INTO frames(hash, frame)
                        VALUES(?,?)''', (hash, frame))
  db.commit()

feats = np.array([ data['videos'][v][frame] for v in data['videos'].keys() for frame in data['videos'][v].keys() ])

n, d = feats.shape

print("processing {}x {}d features...".format(n, d))
index = faiss.index_factory(d, factory_type)

train_start = time.time()
index.train(feats)
train_end = time.time()
train_time = train_end - train_start
print("train time: {}".format(train_time))

add_start = time.time()
# for i in range(10):
#   feats[:, 0] += np.arange(feats) / 1000.
index.add(feats)
add_end = time.time()
add_time = add_end - add_start
print("add time: {}".format(add_time))

# print(index.is_trained)
# print(index.ntotal)

if opt.store_index:
  faiss.write_index(index, index_fn)

keys = list(data['videos'].keys())
key = keys[0]

frames = list(data['videos'][key].keys())
frame = frames[0]

# print(key, frame)

vec = data['videos'][key][frame]

# print('sanity check:')

search_start = time.time()
D, I = index.search(np.array([vec]).astype('float32'), 15)
search_end = time.time()
search_time = search_end - search_start
print("search time: {}".format(search_time))

with open("timing.txt", "a") as f:
  index_size = os.path.getsize(index_fn)
  print("index size: {}".format(int(index_size/(1024*1024))))
  writer = csv.writer(f)
  writer.writerow("#dataset, #index.ntotal, #d, #factory_type, #train_time, #add_time, #search_time, #index_size")
  writer.writerow([dataset, index.ntotal, d, factory_type, train_time, add_time, search_time, index_size])

# print("distances:")
# print(D[0])

# print("indexes:")
# print(I[0])

