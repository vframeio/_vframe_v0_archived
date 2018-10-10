import sys
import os
from os.path import join
import cv2 as cv
import imagehash
from PIL import Image, ImageDraw
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage import feature
import matplotlib.pyplot as plt
import imutils
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from sklearn.metrics.pairwise import cosine_similarity
# paths = ['/media/adam/ah3tb/work/undisclosed/vframe/3rdparty/img2vec/',
#           '/media/adam/adamt2b/work/undisclosed/vframe/3rdparty/img2vec/']
# for p in paths:
#   if os.path.isdir(p):
#     print(p)
#     sys.path.append(p)
# from img_to_vec import Img2Vec

# img2vec = Img2Vec(cuda=True)

def compute_velocity(vals):
  """returns np.array of difference v2-v1, with i0 = 0"""
  vels = np.zeros_like(vals)
  vels[0] = 0 # first frame has no velocity
  for i,v in enumerate(vals[1:]):
    vels[i] = abs(v - vals[i-1])
  return vels

def ensure_pil(im):
  """Ensure image is Pillow format"""
  try:
      im.verify()
      return im
  except:
      return Image.fromarray(im.astype('uint8'), 'RGB')

def ensure_np(im):
  """Ensure image is numpy array"""
  if type(im) == np.ndarray:
      return im
  return np.asarray(im, np.uint8)

def ensure_dir(d):
  """Create directories"""
  if not os.path.exists(d):
      os.makedirs(d)

def filter_pixellate(src,num_cells):
  """Downsample, then upsample image for pixellation"""
  w,h = src.size
  dst = src.resize((num_cells,num_cells), Image.NEAREST)
  dst = dst.resize((w,h), Image.NEAREST)
  return dst

# Plot images inline using Matplotlib
def pltimg(im,title=None,mode='rgb',figsize=(8,12),dpi=160,output=None):
  plt.figure(figsize=figsize)
  plt.xticks([]),plt.yticks([])
  if title is not None:
      plt.title(title)
  if mode.lower() == 'bgr':
      im = cv.cvtColor(im,cv.COLOR_BGR2RGB)

  f = plt.gcf()
  if mode.lower() =='grey' or mode.lower() == 'gray':
    plt.imshow(im,cmap='gray')
  else:
      plt.imshow(im)
  plt.show()
  plt.draw()
  if output is not None:
      bbox_inches='tight'
      ext=osp.splitext(output)[1].replace('.','')
      f.savefig(output,dpi=dpi,format=ext)
      print('Image saved to: {}'.format(output))


# Define a function to detect faces using OpenCV's haarcascades
def detect_faces(classifier,src,scale_factor=1.1,overlaps=3,
                min_size=70, max_size=700,
                flags=0):
  
  min_size = (min_size, min_size) # minimum face size
  max_size = (max_size, max_size) # maximum face size
  
  # Convert to grayscale
  src_gray = cv.cvtColor(src,cv.COLOR_BGR2GRAY)
  
  # Run detector
  matches = classifier.detectMultiScale(src_gray, 
                                        scale_factor, 
                                        overlaps, 
                                        flags, 
                                        min_size, 
                                        max_size)
  # By default, this returns x,y,w,w
  # Modify to return x1,y1,x2,y2
  matches = [ (r[0],r[1],r[0]+r[2],r[1]+r[3]) for r in matches]
  
  return matches

def detect_faces_dlib(im,pyramids=0):
  rects = detector(im, pyramids)
  faces = [ (r.left(),r.top(),r.right(),r.bottom()) for r in rects] #x1,y1,x2,y2
  return faces


# Utilities for analyzing frames

def compute_gray(im):
  im = cv.cvtColor(im,cv.COLOR_BGR2GRAY)
  n_vals = float(im.shape[0] * im.shape[1])
  avg = np.sum(im[:]) / n_vals
  return avg

def compute_rgb(im):
  im = cv.cvtColor(im,cv.COLOR_BGR2RGB)
  n_vals = float(im.shape[0] * im.shape[1])
  avg_r = np.sum(im[:,:,0]) / n_vals 
  avg_g = np.sum(im[:,:,1]) / n_vals
  avg_b = np.sum(im[:,:,2]) / n_vals
  avg_rgb = np.sum(im[:,:,:]) / (n_vals * 3.0)
  return avg_r, avg_b, avg_g, avg_rgb

def compute_hsv(im):
  im = cv.cvtColor(im,cv.COLOR_BGR2HSV)
  n_vals = float(im.shape[0] * im.shape[1])
  avg_h = np.sum(frame[:,:,0]) / n_vals
  avg_s = np.sum(frame[:,:,1]) / n_vals
  avg_v = np.sum(frame[:,:,2]) / n_vals
  avg_hsv = np.sum(frame[:,:,:]) / (n_vals * 3.0)
  return avg_h, avg_s, avg_v, avg_hsv

def pys_dhash(im, hashSize=8):
  # resize the input image, adding a single column (width) so we
  # can compute the horizontal gradient
  resized = cv.resize(im, (hashSize + 1, hashSize))
  # compute the (relative) horizontal gradient between adjacent
  # column pixels
  diff = resized[:, 1:] > resized[:, :-1]
  # convert the difference image to a hash
  return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

def ahash(im_np):
  # average
  im_pil = ensure_pil(im_np)
  return imagehash.average_hash(im_pil)

def phash(im_np):
  # perceptual
  im_pil = ensure_pil(im_np)
  return imagehash.phash(im_pil)

def dhash(im_np):
  # difference
  im_pil = ensure_pil(im_np)
  return imagehash.dhash(im_pil)

def whash(im_np):
  # haar wavelet
  im_pil = ensure_pil(im_np)
  return imagehash.whash(im_pil)

def whash_b64(im_np):
  im_pil = ensure_pil(im_np)
  # hashfunc = lambda img: imagehash.whash(img, mode='db4')
  return lambda im_pil: imagehash.whash(im_pil, mode='db4')

def compute_entropy(im):
  entr_img = entropy(im, disk(10))

def bgr2gray(im):
  return cv.cvtColor(im,cv.COLOR_BGR2GRAY)

def compute_entropy(im):
  # im is grayscale numpy
  return entropy(im, disk(10))

def compute_laplacian(im):
  # below 100 is usually blurry
  return cv.Laplacian(im, cv.CV_64F).var()

def compute_if_blank(im,width=100,sigma=0,thresh_canny=.1,thresh_mean=3):
  # im is graysacale np
  im = imutils.resize(im,width=width)
  im_canny = feature.canny(im,sigma=sigma)
  total = (im.shape[0]*im.shape[1])
  n_white = len(np.where(im_canny > 0)[0])
  per = n_white/total
  gray_mean = np.mean(im)
  if np.mean(im) < thresh_mean and per < thresh_canny:
    return 1
  else:
    return 0

def print_timing(t,n):
    t = time.time()-t
    print('Elapsed time: {:.2f}'.format(t))
    print('FPS: {:.2f}'.format(n/t))

def vid2frames(fpath,limit=9999999,width=None):
  frames = []
  cap = cv.VideoCapture(fpath)
  while(True and len(frames) < limit):
    res, frame = cap.read()
    if not res:
      cap.release()
      break
    if width is not None:
      frame = imutils.resize(frame,width=width)
    frames.append(frame)
  return frames

def convolve_filter(vals,filters=[1]):
  for k in filters:
    vals_tmp = np.zeros_like(vals)
    t = len(vals_tmp)
    for i,v in enumerate(vals):
      sum_vals = vals[max(0,i-k):min(t-1,i+k)]
      vals_tmp[i] = np.mean(sum_vals)
    vals = vals_tmp.copy()
  return vals

def cosine_delta(v1,v2):
  return cosine_similarity(v1.reshape((1, -1)), v2.reshape((1, -1)))[0][0]


# http://radjkarl.github.io/imgProcessor/index.html#

def modifiedLaplacian(img):
    ''''LAPM' algorithm (Nayar89)'''
    M = np.array([-1, 2, -1])
    G = cv.getGaussianKernel(ksize=3, sigma=-1)
    Lx = cv.sepFilter2D(src=img, ddepth=cv.CV_64F, kernelX=M, kernelY=G)
    Ly = cv.sepFilter2D(src=img, ddepth=cv.CV_64F, kernelX=G, kernelY=M)
    FM = np.abs(Lx) + np.abs(Ly)
    return cv.mean(FM)[0]

  
def varianceOfLaplacian(img):
    ''''LAPV' algorithm (Pech2000)'''
    lap = cv.Laplacian(img, ddepth=-1)#cv.cv.CV_64F)
    stdev = cv.meanStdDev(lap)[1]
    s = stdev[0]**2
    return s[0]

def tenengrad(img, ksize=3):
    ''''TENG' algorithm (Krotkov86)'''
    Gx = cv.Sobel(img, ddepth=cv.CV_64F, dx=1, dy=0, ksize=ksize)
    Gy = cv.Sobel(img, ddepth=cv.CV_64F, dx=0, dy=1, ksize=ksize)
    FM = Gx**2 + Gy**2
    return cv.mean(FM)[0]


def normalizedGraylevelVariance(img):
    ''''GLVN' algorithm (Santos97)'''
    mean, stdev = cv.meanStdDev(img)
    s = stdev[0]**2 / mean[0]
    return s[0]


def compute_endpoints(vals,idxs,pos_neg,vel_min=2.5,duration_min=.25,duration_max=2.5,fps=30):
  """Find endpoints and determine whether rising or falling"""
  # idx are either rising(1) or falling(-1)
  endpoints = []
  t = len(vals)
  count_max = fps * duration_max
  count_min = fps * duration_min
  for i,idx in enumerate(idxs):
    count = 1
    if pos_neg == 1:
      vel = abs(vals[idx+count])
      while vel >= vel_min and count < count_max and count < t-1:
        vel = abs(vals[idx+count])
        count += 1
      if count > count_min and count < count_max:
        endpoints.append((idx,idx+(count*pos_neg)))
    elif pos_neg == -1:
      vel = abs(vals[idx+(count*pos_neg)])
      while vel >= vel_min and count < count_max and t-count >= 0:
        vel = abs(vals[idx+(count*pos_neg)])
        count += 1
      if count > count_min and count < count_max:
        endpoints.append((idx,idx+(count*pos_neg)))
  return endpoints


def compute_edges(vals):
  # find edges (1 = rising, -1 = falling)
  edges = np.zeros_like(vals)
  for i in range(len(vals[1:])):
    delta = vals[i] - vals[i-1]
    if delta == -1:
      edges[i] = 1 # rising edge 0 --> 1
    elif delta == 1:
      edges[i+1] = 2 # falling edge 1 --> 0
  # get index for rise fall
  rising = np.where(np.array(edges) == 1)[0]
  falling = np.where(np.array(edges) == 2)[0]
  return rising, falling 