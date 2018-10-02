#!python

import os
import faiss
import pickle
import numpy as np
import sqlite3
import time
import csv
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--factory_type', required=True)
parser.add_argument('--store_db', action='store_true')
parser.add_argument('--store_index', action='store_true')
opt = parser.parse_args()

dataset = opt.dataset
factory_type = opt.factory_type

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

