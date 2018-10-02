# coding: utf-8
import os, sys
from os.path import join
import cv2 as cv
import time
import numpy as np
import imutils
from PIL import Image
from skimage import feature
from random import randint
sys.path.append('/vframe/core/app')
from utils import imx # image utils
from utils import fiox # file utils
from config import config as cfg
import imagehash


def find_logo_regions(cfg,f):

  # read sample indices by seeking to frame index
  frames_orig = []
  phashes = []
  for idx in idxs:
    cap.set(cv.CAP_PROP_POS_FRAMES, idx)
    res, frame = cap.read()
    phash = imx.phash(imutils.resize(frame,width=160))
    # check all previous frames if phash is same
    found_dupe = False
    for p in phashes:
      if abs(phash - p) < cfg.phash_thresh:
        found_dupe = True
        break
    if not found_dupe:
      frames_orig.append(frame)
      phashes.append(phash)

  if len(frames_orig) < cfg.min_num_samples:
    print('[-] Too few samples, probably')
    print('[-] Skipping: {}'.format(bname))
    return
  else:
    ndupes = len(idxs) - len(frames_orig)
    print('[+] Found {} duplicate frames / {}'.format(ndupes,len(idxs)))
    nsamples = len(frames_orig)
  # remove any duplicate frames
  # Create image pyramids
  frames = [imutils.resize(f,width=cfg.image_resolution) for f in frames_orig]
  
  # create edge images (~7FPS on 960px width images)
  edge_padding_per = 0.01

  # randomly sample frames and compute average of canny/edges in grayscale
  # important to choose a high-resolution frame to get good edges
  st = time.time()
  # mask edges slightly, some videos have edge jitter
  mask = np.zeros_like(imx.bgr2gray(frames[0])).astype(np.bool)
  h,w = frames[0].shape[:2]
  margin = int(edge_padding_per*min(w,h))
  mask[margin:h-margin,margin:w-margin] = 1
  #im_samples = [feature.canny(imx.bgr2gray(imx.sharpen(frames[idx])),sigma=0,mask=mask) for idx in range(nsamples)]
  # canny is quicker, but conveys different information, or the threshold needs to be better tuned
  im_samples = [cv.Canny(imx.sharpen(frames[idx]),cfg.canny_min,cfg.canny_max,cfg.canny_size) for idx in range(nsamples)]

  median_width = .375
  median_height = .375
  x1 = int(( (.5-(median_width/2)) * w ))
  x2 = int(( (.5+(median_width/2)) * w ))
  no_logo_rv = (x1,0,x2,h)
  y1 = int(( (.5-(median_height/2)) * h ))
  y2 = int(( (.5+(median_height/2)) * h ))
  no_logo_rh = (0,y1,w,y2)

  # add zone in middle
  cbox_width = .5
  cbox_height = .55
  x1 = int(( (.5-(cbox_width/2)) * w ))
  x2 = int(( (.5+(cbox_width/2)) * w ))
  y1 = int(( (.5-(cbox_height/2)) * h ))
  y2 = int(( (.5+(cbox_height/2)) * h ))
  no_logo_c = (x1,y1,x2,y2)
  no_logo_regions = [no_logo_rh, no_logo_rv,no_logo_c]

  # Filter again to keep only the largest bbox in each quad region
  q1 = (0,0,no_logo_rv[0],no_logo_rh[1])
  q2 = (no_logo_rv[2],0,w-1,no_logo_rh[1])
  q3 = (0,no_logo_rh[3],no_logo_rv[0],h-1)
  q4 = (no_logo_rv[2],no_logo_rh[3],w-1,h-1)
  quad_regions = [q1,q2,q3,q4]

  # remove sampled images that don't contain logo-corner information
  #im_samples_temp = im_samples.copy()
  #im_samples = [] # clear
  #for i,im in enumerate(im_samples_temp):
  #  for x1,y1,x2,y2 in no_logo_regions:
  #    im[y1:y2,x1:x2] = 0
  #  if np.mean(im) > 1:
  #    im_samples.append(im)
  #imx.print_timing(st,len(im_samples))

  if not len(im_samples) > 0:
    print('[-] Skipping {} (no edge information)'.format(bname))
    return
  # create canny average
  #im_canny_mean = (255*np.mean(im_samples,axis=0)).astype(np.uint8) # sklean
  im_canny_mean = np.mean(im_samples,axis=0).astype(np.uint8) # cv canny
  
  #im_canny_mean_orig = im_canny_mean.copy() # for debugging

  # mask edges slightly, some videos have edge jitter around edges7
  h,w = im_canny_mean.shape[:2]
  margin = int(edge_padding_per*min(w,h))
  im_canny_mean_mask = np.zeros_like(im_canny_mean)
  im_canny_mean_mask[margin:h-margin,margin:w-margin] = im_canny_mean[margin:h-margin,margin:w-margin]
  im_canny_mean = im_canny_mean_mask

  # check for cropped video by scanning for horizontal or vertical black lines

  bin_inc = cfg.bin_thresh_inc
  bin_var = 99999
  bin_thresh_min = cfg.bin_thresh_min
  bin_thresh_max = cfg.bin_thresh_max
  bin_thresh = bin_thresh_min

  # measure only information in quad-zones (logo areas)
  while bin_var > cfg.bin_var_thresh and bin_thresh < bin_thresh_max:
    var_max = 0
    for q_idx,region in enumerate(quad_regions):
      x1,y1,x2,y2 = region
      im_quad_edge = im_canny_mean[y1:y2,x1:x2].copy()
      #im_canny_thresh = im_canny_mean.copy()
      im_quad_edge[im_quad_edge < bin_thresh] = 0
      im_quad_edge[im_quad_edge > bin_thresh] = 255
      cur_var = im_quad_edge.var()
      if cur_var > var_max:
        bin_var = cur_var
        var_max = bin_var
      #print('[+] {} binary variance: {:.2f}, max: {}, thresh: {}'.format(q_idx, cur_var,var_max, bin_thresh))
    bin_thresh += bin_inc

  bin_thresh -= bin_inc
  im_canny_thresh = im_canny_mean.copy()
  im_canny_thresh[im_canny_mean < bin_thresh] = 0
  im_canny_thresh[im_canny_mean > bin_thresh] = 255
  #bin_var = im_canny_thresh.var()
  print('[+] final binary variance: {:.2f}, thresh: {}'.format(bin_var,bin_thresh))

  # dilate and erod
  im_canny_optim = im_canny_thresh.copy()
  kernel = np.ones((3,3),np.uint8)
  im_canny_optim = cv.morphologyEx(im_canny_optim, cv.MORPH_CLOSE, kernel, iterations = 1) # infill

  kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (1,1)) # remove noise
  im_canny_optim = cv.erode(im_canny_optim, kernel, iterations = 2)
  
  kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
  im_canny_optim = cv.dilate(im_canny_optim, kernel, iterations = 2) # infill

  kernel = np.ones((3,3),np.uint8)
  im_canny_optim = cv.morphologyEx(im_canny_optim, cv.MORPH_CLOSE, kernel, iterations = 1) # infill
  #im_canny_optim = cv.blur(im_canny_optim,(3,3))
  
  kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (1,1)) # remove noise
  im_canny_optim = cv.erode(im_canny_optim, kernel, iterations = 1)

  max_area = int((cfg.max_logo_size[0]*w) * (cfg.max_logo_size[1]*h))
  min_area = int((cfg.min_logo_size[0]*w) * (cfg.min_logo_size[1]*h))

  # delta = 5
  # #max_variation = 0.25;
  # max_variation = 0.5;
  # min_diversity = .2;
  # max_evolution = 200;
  # area_threshold = 1.01;
  # min_margin = .003;
  # edge_blur_size = 5;

  #im_canny_optim = cv.blur(im_canny_optim,(3,3))
  # maximally stable region
  #mser = cv.MSER_create(delta,min_area,max_area,max_variation,min_diversity,max_evolution,area_threshold,min_margin,edge_blur_size)
  mser = cv.MSER_create()
  #MSER mser = MSER.create(delta, min_area, max_area, max_variation, min_diversity, max_evolution, area_threshold, min_margin, edge_blur_size);

  print('min mser: {}, max mser: {}'.format(min_area,max_area))
  mser.setMaxArea(max_area)
  mser.setMinArea(min_area)

  #mser.setDelta(5) # what does delta do?
  regions, bboxes = mser.detectRegions(im_canny_optim)
  
  # merge overlapping boxes
  bboxes = [[x,y,w+x,h+y] for x,y,w,h in bboxes]
  h,w = im_canny_optim.shape[:2]
  total_pixels = w*h

  print('mser bboxes raw: {}'.format(len(bboxes)))
  print('mser bboxes regions: {}'.format(len(regions)))

  # remove anythin in the no-logo zone
  bboxes = [bbox for bbox in bboxes if \
    not imx.is_overlapping(bbox,no_logo_rh) and \
    not imx.is_overlapping(bbox,no_logo_rv)]

  print('bboxes after xoverlapping: {}'.format(len(bboxes)))
  # merge overlapping boxes with optional expanding region
  bboxes_merged = set(imx.get_rects_merged(bboxes,(w,h),expand=cfg.expand))

  # TODO this removes duplicates after mergingn, but should be one function
  bboxes_merged = set(imx.get_rects_merged(bboxes,(w,h),expand=cfg.expand))
  
  print('bboxes merged: {}'.format(len(bboxes_merged)))
  # remove any boxes that are too horizontal or verticle 
  bboxes_merged = [[x1,y1,x2,y2] for x1,y1,x2,y2 in bboxes_merged \
    if (x2-x1)//(y2-y1) < cfg.max_ratio_wh and \
    (y2-y1)//(x2-x1) < cfg.max_ratio_hw] # filter out odd sizes
  
  # filter final size
  for x1,y1,x2,y2 in bboxes_merged:
    per = ((x2-x1)*(y2-y1))/total_pixels
    print('{:.4f}'.format(per))

  bboxes_merged = [[x1,y1,x2,y2] for x1,y1,x2,y2 in bboxes_merged \
   if ((x2-x1)*(y2-y1))/total_pixels > cfg.min_final_area and \
   ((x2-x1)*(y2-y1))/total_pixels < cfg.max_final_area] # filter out extreme sizes

  quads = [{'region':r,'area':0,'bbox':[0,0,0,0]} for r in quad_regions]

  for bbox in bboxes_merged:
    x1,y1,x2,y2 = bbox
    bw,bh = (x2-x1,y2-y1)
    bbox_area = bw*bh
    for i in range(len(quads)):
      quad = quads[i]
      if imx.is_overlapping(bbox,quad['region']):
        quad_area = quad['area']
        if quad_area == 0 or bbox_area > quad_area:
          quad['bbox'] = bbox
          quad['area'] = bbox_area

  bboxes_merged = [] # clear
  bboxes_merged = [quad['bbox'] for quad in quads if quad['area'] > 0]


  result = {
      'bboxes':bboxes_merged,
      'image_mean':im_canny_mean,
      'image_optim':im_canny_optim,
      'image_thresh':im_canny_thresh,
      }

  return result