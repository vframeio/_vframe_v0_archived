#!python

import os
from os import path
import pickle
from db import FaissSearch
import numpy as np

name = 'alexnet'

base_path = path.dirname(path.abspath(__file__))
pkl_fn = path.join(base_path, "features", name, "index.pkl")

fh = open(pkl_fn, 'rb')
raw = fh.read()
fh.close()
data = pickle.loads(raw)

keys = list(data['videos'].keys())
key = keys[0]
frames = list(data['videos'][key].keys())
frame = frames[0]
vec = data['videos'][key][frame]
vec = np.array([vec])

with open("test.html", "w") as f:
  f.write('<link rel="stylesheet" type="text/css" href="test.css" />')
  for fn in os.listdir('indexes'):
    if '.index' not in fn:
      continue
    print(fn)
    faiss_db = FaissSearch(name, fn.replace('.index', ''))
    results = faiss_db.search(vec, limit=10)
    # print(results)

    f.write('<h3>{}</h3>'.format(fn))
    f.write('<div class="result">')
    for result in results:
      hash = result['hash']
      frame = result['frame']
      score = "{0:.2f}".format(round(result['distance'], 2))
      url = "https://sa-vframe.ams3.digitaloceanspaces.com/v1/media/frames/{}/{}/{}/{}/{}/md/index.jpg".format(hash[0:3], hash[3:6], hash[6:9], hash, frame)
      f.write('<div><img src="{}"><span>{}</span></div>'.format(url, score))
    f.write('</div>')
