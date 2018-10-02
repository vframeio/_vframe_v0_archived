#!python

from os import path
import faiss
import pickle
import numpy as np
import sqlite3

class FaissSearch:
  def __init__(self, name, index_name=None):
    base_path = path.dirname(path.abspath(__file__))
    db_fn = path.join(base_path, "indexes", "{}.db".format(name))
    index_fn = path.join(base_path, "indexes", "{}.index".format(index_name if index_name is not None else name))

    self.db = sqlite3.connect(db_fn)
    self.index = faiss.read_index(index_fn)
    # print(self.index)

  def search(self, query, limit=15):
    cursor = self.db.cursor()

    # D = distances, I = indexes
    D, I = self.index.search(query, limit)

    # print("distances:")
    # print(D[0])

    # print("indexes:")
    # print(I[0])

    lookup = {}
    for d, i in zip(D[0], I[0]):
      lookup[i+1] = d

    q = '''
        SELECT id, hash, frame FROM frames WHERE id IN {}
    '''.format(tuple([i+1 for i in I[0]]))
    cursor.execute(q)

    rows = cursor.fetchall()
    # print(rows)

    results = []
    for row in rows:
      _id = row[0]
      if _id in lookup:
        results.append({
          'index': _id,
          'distance': lookup[_id],
          'hash': row[1],
          'frame': row[2],
        })
    return results

if __name__ == '__main__':
  name = 'vgg16'
  faiss_db = FaissSearch(name)

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

  results = faiss_db.search(np.array([vec]))
  print(results)
