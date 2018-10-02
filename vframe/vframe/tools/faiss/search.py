#!python

import sys
import faiss
import cv2 as cv
from ..feature_extractor import FeatureExtractor

fe = FeatureExtractor(net='VGG16', weights='imagenet')
index = faiss.read_index("vgg16.index")

# number of results
LIMIT = 15

img_path = sys.argv[1]

img = cv.imread(img_path)
query = fe.extract(img)

D, I = index.search(query, LIMIT)

# D = distances, I = indexes

print("distances:")
print(D)

print("indexes:")
print(I)
