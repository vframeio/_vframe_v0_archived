import sys
import os
from os.path import join
import cv2 as cv
import imagehash
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage import feature
# import matplotlib.pyplot as plt
import imutils
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from sklearn.metrics.pairwise import cosine_similarity
import datetime




def compute_features(fe,frames,phashes,phash_thresh=1):
  """
  Get vector embedding using FeatureExtractor
  :param fe: FeatureExtractor class
  :param frames: list of frame images as numpy.ndarray
  :param phash_thresh: perceptual hash threshold
  :returns: list of feature vectors
  """
  vals = []
  phash_pre = phashes[0]
  for i,im in enumerate(frames):
    if i == 0 or (phashes[i] - phashes[i-1]) > phash_thresh:
      vals.append(fe.extract(im))
    else:
      vals.append(vals[i-1])
  return vals


def ensure_pil(im, bgr2rgb=False):
  """Ensure image is Pillow format
    :param im: image in numpy or PIL.Image format
    :returns: image in Pillow RGB format
  """
  try:
      im.verify()
      return im
  except:
    if bgr2rgb:
      im = cv.cvtColor(im,cv.COLOR_BGR2RGB)
    return Image.fromarray(im.astype('uint8'), 'RGB')

def ensure_np(im):
  """Ensure image is Numpy.ndarry format
    :param im: image in numpy or PIL.Image format
    :returns: image in Numpy uint8 format
  """
  if type(im) == np.ndarray:
      return im
  return np.asarray(im, np.uint8)


def resize(im,width=0,height=0):
  """resize image using imutils. Use w/h=[0 || None] to prioritize other edge size
    :param im: a Numpy.ndarray image
    :param wh: a tuple of (width, height)
  """
  w = width
  h = height
  if w is 0 and h is 0:
    return im
  elif w > 0 and h > 0:
    return imutils.resize(im,width=w,height=h)
  elif w > 0 and h is 0:
    return imutils.resize(im,width=w)
  elif w is 0 and h > 0:
    return imutils.resize(im,height=h)
  else:
    return im

def filter_pixellate(im,num_cells):
  """Pixellate image by downsample then upsample
    :param im: PIL.Image
    :returns: PIL.Image
  """
  w,h = im.size
  im = im.resize((num_cells,num_cells), Image.NEAREST)
  im = im.resize((w,h), Image.NEAREST)
  return im

# Plot images inline using Matplotlib
# def pltimg(im,title=None,mode='rgb',figsize=(8,12),dpi=160,output=None):
#   plt.figure(figsize=figsize)
#   plt.xticks([]),plt.yticks([])
#   if title is not None:
#     plt.title(title)
#   if mode.lower() == 'bgr':
#     im = cv.cvtColor(im,cv.COLOR_BGR2RGB)

#   f = plt.gcf()
#   if mode.lower() =='grey' or mode.lower() == 'gray':
#     plt.imshow(im,cmap='gray')
#   else:
#     plt.imshow(im)
#   plt.show()
#   plt.draw()
#   if output is not None:
#     bbox_inches='tight'
#     ext=osp.splitext(output)[1].replace('.','')
#     f.savefig(output,dpi=dpi,format=ext)
#     print('Image saved to: {}'.format(output))



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


############################################
# ImageHash 
# pip install imagehash
############################################


def compute_ahash(im):
  """Compute average hash using ImageHash library
    :param im: Numpy.ndarray
    :returns: Imagehash.ImageHash
  """
  return imagehash.average_hash(ensure_pil(im_pil))

def compute_phash(im):
  """Compute perceptual hash using ImageHash library
    :param im: Numpy.ndarray
    :returns: Imagehash.ImageHash
  """
  return imagehash.phash(ensure_pil(im))

def compute_dhash(im):
  """Compute difference hash using ImageHash library
    :param im: Numpy.ndarray
    :returns: Imagehash.ImageHash
  """
  return imagehash.dhash(ensure_pil(im))

def compute_whash(im):
  """Compute wavelet hash using ImageHash library
    :param im: Numpy.ndarray
    :returns: Imagehash.ImageHash
  """
  return imagehash.whash(ensure_pil(im))

def compute_whash_b64(im):
  """Compute wavelest hash base64 using ImageHash library
    :param im: Numpy.ndarray
    :returns: Imagehash.ImageHash
  """
  return lambda im: imagehash.whash(ensure_pil(im), mode='db4')


############################################
# Pillow 
############################################

def sharpen(im):
  """Sharpen image using PIL.ImageFilter
  param: im: PIL.Image
  returns: PIL.Image
  """
  im = ensure_pil(im)
  im.filter(ImageFilter.SHARPEN)
  return ensure_np(im)

def fit_image(im,targ_size):
  """Force fit image by cropping
  param: im: PIL.Image
  param: targ_size: a tuple of target (width, height)
  returns: PIL.Image
  """
  im_pil = ensure_pil(im)
  frame_pil = ImageOps.fit(im_pil, targ_size, 
    method=Image.BICUBIC, centering=(0.5, 0.5))
  return ensure_np(frame_pil)


def compute_entropy(im):
  entr_img = entropy(im, disk(10))


############################################
# scikit-learn 
############################################

def compute_entropy(im):
  # im is grayscale numpy
  return entropy(im, disk(10))

############################################
# OpenCV 
############################################

def bgr2gray(im):
  """Wrapper for cv2.cvtColor transform
    :param im: Numpy.ndarray (BGR)
    :returns: Numpy.ndarray (Gray)
  """
  return cv.cvtColor(im,cv.COLOR_BGR2GRAY)

def gray2bgr(im):
  """Wrapper for cv2.cvtColor transform
    :param im: Numpy.ndarray (Gray)
    :returns: Numpy.ndarray (BGR)
  """
  return cv.cvtColor(im,cv.COLOR_GRAY2BGR)

def bgr2rgb(im):
  """Wrapper for cv2.cvtColor transform
    :param im: Numpy.ndarray (BGR)
    :returns: Numpy.ndarray (RGB)
  """
  return cv.cvtColor(im,cv.COLOR_BGR2RGB)

def compute_laplacian(im):
  # below 100 is usually blurry
  return cv.Laplacian(im, cv.CV_64F).var()


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

def compute_if_blank(im,width=100,sigma=0,thresh_canny=.1,thresh_mean=4,mask=None):
  # im is graysacale np
  #im = imutils.resize(im,width=width)
  #mask = imutils.resize(mask,width=width)
  if mask is not None:
    im_canny = feature.canny(im,sigma=sigma,mask=mask)
    total = len(np.where(mask > 0)[0])
  else:
    im_canny = feature.canny(im,sigma=sigma)
    total = (im.shape[0]*im.shape[1])
  n_white = len(np.where(im_canny > 0)[0])
  per = n_white/total
  if np.mean(im) < thresh_mean or per < thresh_canny:
    return 1
  else:
    return 0


def print_timing(t,n):
    t = time.time()-t
    print('Elapsed time: {:.2f}'.format(t))
    print('FPS: {:.2f}'.format(n/t))

def vid2frames(fpath, limit=5000, width=None, idxs=None):
  """Convert a video file into list of frames
    :param fpath: filepath to the video file
    :param limit: maximum number of frames to read
    :param fpath: the indices of frames to keep (rest are skipped)
    :returns: (fps, number of frames, list of Numpy.ndarray frames)
  """
  frames = []
  try:
    cap = cv.VideoCapture(fpath)
  except:
    print('[-] Error. Could not read video file: {}'.format(fpath))
    try:
      cap.release()
    except:
      pass
    return frames

  fps = cap.get(cv.CAP_PROP_FPS)
  nframes = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

  if idxs is not None:
    # read sample indices by seeking to frame index
    for idx in idxs:
      cap.set(cv.CAP_PROP_POS_FRAMES, idx)
      res, frame = cap.read()
      if width is not None:
        frame = imutils.resize(frame, width=width)
      frames.append(frame)
  else:
    while(True and len(frames) < limit):
      res, frame = cap.read()
      if not res:
        break
      if width is not None:
        frame = imutils.resize(frame, width=width)
      frames.append(frame)

  cap.release()
  del cap
  #return fps,nframes,frames
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
  return 1.0 - cosine_similarity(v1.reshape((1, -1)), v2.reshape((1, -1)))[0][0]



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


############################################
# Point, Rect
############################################

class Point(object):
  def __init__(self, x, y):
    self.x = x
    self.y = y

class Rect(object):
  def __init__(self, p1, p2):
    '''Store the top, bottom, left and right values for points 
           p1 and p2 are the (corners) in either order
    '''
    self.left   = min(p1.x, p2.x)
    self.right  = max(p1.x, p2.x)
    self.top    = min(p1.y, p2.y)
    self.bottom = max(p1.y, p2.y)
    
def overlap(r1, r2):
    '''Overlapping rectangles overlap both horizontally & vertically
    '''
    return range_overlap(r1.left, r1.right, r2.left, r2.right) and \
      range_overlap(r1.top, r1.bottom, r2.top, r2.bottom)

def range_overlap(a_min, a_max, b_min, b_max):
  '''Neither range is completely greater than the other
  '''
  return (a_min <= b_max) and (b_min <= a_max)

def merge_rects(r1,r2):
  p1 = Point(min(r1.left,r2.left),min(r1.top,r2.top))
  p2 = Point(max(r1.right,r2.right),max(r1.bottom,r2.bottom))
  return Rect(p1,p2)

def is_overlapping(r1,r2):
  """r1,r2 as [x1,y1,x2,y2] list"""
  r1x = Rect(Point(r1[0],r1[1]),Point(r1[2],r1[3]))
  r2x = Rect(Point(r2[0],r2[1]),Point(r2[2],r2[3]))
  return overlap(r1x,r2x)

def get_rects_merged(rects,bounds,expand=0):
  """rects: list of points in [x1,y1,x2,y2] format"""
  rects_expanded = []
  bx,by = bounds
  # expand
  for x1,y1,x2,y2 in rects:
    x1 = max(0,x1-expand)
    y1 = max(0,y1-expand)
    x2 = min(bx,x2+expand)
    y2 = min(by,y2+expand)
    rects_expanded.append(Rect(Point(x1,y1),Point(x2,y2)))

  #rects_expanded = [Rect(Point(x1,y1),Point(x2,y2)) for x1,y1,x2,y2 in rects_expanded]
  rects_merged = []
  for i,r in enumerate(rects_expanded):
    found = False
    for j,rm in enumerate(rects_merged):
      if overlap(r,rm):
        rects_merged[j] = merge_rects(r,rm) #expand
        found = True
    if not found:
      rects_merged.append(r)
  # convert back to [x1,y1,x2,y2] format
  rects_merged = [(r.left,r.top,r.right,r.bottom) for r in rects_merged]
  # contract
  rects_contracted = []
  for x1,y1,x2,y2 in rects_merged:
    x1 = min(bx,x1+expand)
    y1 = min(by,y1+expand)
    x2 = max(0,x2-expand)
    y2 = max(0,y2-expand)
    rects_contracted.append((x1,y1,x2,y2))

  return rects_contracted


############################################
# Image display
############################################


def montage(frames,ncols=4,nrows=None,width=None):
  """Convert list of frames into a grid montage
  param: frames: list of frames as Numpy.ndarray
  param: ncols: number of columns
  param: width: resize images to this width before adding to grid
  returns: Numpy.ndarray grid of all images
  """

  # expand image size if not enough frames
  if nrows is not None and len(frames) < ncols * nrows:
    blank = np.zeros_like(frames[0])
    n = ncols * nrows - len(frames)
    for i in range(n): frames.append(blank) 

  rows = []
  for i,im in enumerate(frames):
    if width is not None:
      im = imutils.resize(im,width=width)
    h,w = im.shape[:2]
    if i % ncols == 0:
      if i > 0:
        rows.append(ims)
      ims = []
    ims.append(im)
  if len(ims) > 0:
    for j in range(ncols-len(ims)):
      ims.append(np.zeros_like(im))
    rows.append(ims)
  row_ims = []
  for row in rows:
    row_im = np.hstack(np.array(row))
    row_ims.append(row_im)
  contact_sheet = np.vstack(np.array(row_ims))
  return contact_sheet
