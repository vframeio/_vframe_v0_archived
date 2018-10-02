
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
from logo_finder import find_logo_regions

def main(args):
  
  if os.path.isdir(args.input):
    from glob import glob
    files = glob(join(args.input,'*.mp4'))
  else:
    files = [args.input]

  if len(files) == 0:
    print('No files in "{}"'.format(args.input))
    sys.exit()

  # create directories where needed
  if args.save_debug:
    dir_debug = join(args.output,'debug')
    fiox.ensure_dir(dir_debug)

  if not args.no_save:
    fiox.ensure_dir(args.output)

  if not args.no_save_log:
    # write to log file
    dir_log = join(args.output,'log')
    fiox.ensure_dir(dir_log)
    f_logfile = join(dir_log,'log.csv')
    # start file
    with open(f_logfile,'w') as fp:
      fp.write('# filename, fps, frames, duplicates, logos_found\n')

  for f in files:
    print('Process {}'.format(os.path.basename(f)))
    if os.path.isfile(f):
      extract_logo(args,f)


def extract_logo(args,f):

  bname = os.path.basename(f)
  name,ext = os.path.splitext(bname)

  # load video
  cap = cv.VideoCapture(f)
  fps = cap.get(cv.CAP_PROP_FPS)
  nframes = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
  nsamples = min(int(args.sample_size*fps),nframes-1)

  # get random sample indices
  #idxs = np.random.choice(range(nframes),nsamples,replace=False)
  sample_spacing = nframes//nsamples
  idxs = np.arange(0,nframes,sample_spacing)
  idxs.sort()

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
      if abs(phash - p) < args.phash_thresh:
        found_dupe = True
        break
    if not found_dupe:
      frames_orig.append(frame)
      phashes.append(phash)

  if len(frames_orig) < args.min_num_samples:
    print('[-] Too few samples, probably')
    print('[-] Skipping: {}'.format(bname))
    return
  else:
    ndupes = len(idxs) - len(frames_orig)
    print('[+] Found {} duplicate frames / {}'.format(ndupes,len(idxs)))
    nsamples = len(frames_orig)
  # remove any duplicate frames
  # Create image pyramids
  frames = [imutils.resize(f,width=args.image_resolution) for f in frames_orig]
  
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
  im_samples = [cv.Canny(imx.sharpen(frames[idx]),args.canny_min,args.canny_max,args.canny_size) for idx in range(nsamples)]

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

  bin_inc = args.bin_thresh_inc
  bin_var = 99999
  bin_thresh_min = args.bin_thresh_min
  bin_thresh_max = args.bin_thresh_max
  bin_thresh = bin_thresh_min

  # measure only information in quad-zones (logo areas)
  while bin_var > args.bin_var_thresh and bin_thresh < bin_thresh_max:
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

  max_area = int((args.max_logo_size[0]*w) * (args.max_logo_size[1]*h))
  min_area = int((args.min_logo_size[0]*w) * (args.min_logo_size[1]*h))

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
  bboxes_merged = set(imx.get_rects_merged(bboxes,(w,h),expand=args.expand))

  # TODO this removes duplicates after mergingn, but should be one function
  bboxes_merged = set(imx.get_rects_merged(bboxes,(w,h),expand=args.expand))
  
  print('bboxes merged: {}'.format(len(bboxes_merged)))
  # remove any boxes that are too horizontal or verticle 
  bboxes_merged = [[x1,y1,x2,y2] for x1,y1,x2,y2 in bboxes_merged \
    if (x2-x1)//(y2-y1) < args.max_ratio_wh and \
    (y2-y1)//(x2-x1) < args.max_ratio_hw] # filter out odd sizes
  
  # filter final size
  for x1,y1,x2,y2 in bboxes_merged:
    per = ((x2-x1)*(y2-y1))/total_pixels
    print('{:.4f}'.format(per))

  bboxes_merged = [[x1,y1,x2,y2] for x1,y1,x2,y2 in bboxes_merged \
   if ((x2-x1)*(y2-y1))/total_pixels > args.min_final_area and \
   ((x2-x1)*(y2-y1))/total_pixels < args.max_final_area] # filter out extreme sizes

  #quad_bboxes = [[0,0,0,0] for r in quad_regions] # init with empty rects
  #quad_bboxes_areas = [0 for r in quad_regions] # init with empty rects

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

  print('merged',bboxes_merged)
  bboxes_merged = [] # clear
  bboxes_merged = [quad['bbox'] for quad in quads if quad['area'] > 0]

  print('merged concat',bboxes_merged)

   # grab a random frame and extract logo
  logos = []
  best_idxs = []
  scale = frames_orig[0].shape[1]/frames[0].shape[1]
  if len(bboxes_merged) == 0:
    print('[-] No logos found in: {}'.format(os.path.basename(f)))
  else:
    for i,bbox in enumerate(bboxes_merged):
      # find best frame by searching all sampled frames
      logo_scores = []
      for f in im_samples:
        x1,y1,x2,y2 = [int(b) for b in bbox]
        crop = f[y1:y2,x1:x2]
        # find per white
        logo_scores.append(len(np.where(crop == 255)[0]))

      # Rank logos by edge energy, but take average
      mean_score = np.mean(logo_scores,axis=0)
      best_idx = np.where(logo_scores > mean_score)[0][0]
      best_idxs.append(best_idx)
      ##
      # TODO: partition best N scores and save to disk
      ##

      x1,y1,x2,y2 = [int(scale * b) for b in bbox]
      im = frames_orig[best_idx] # random first frame
      im_logo = im[y1:y2,x1:x2]
      if not args.no_save:
        fpath = join(args.output,'{}_{}.png'.format(name,i))
        print('[+] Save logo {}'.format(fpath))
        cv.imwrite(fpath,im_logo)


  # convert to bgr for drawing
  idx = best_idxs[0] if len(best_idxs) > 0 else randint(0,nsamples-1)
  im_bboxes = frames[idx] # random frame
  im_bboxes_blur = cv.blur(im_bboxes,(args.blur_amt,args.blur_amt))
  im_black = np.zeros_like(im_bboxes_blur)
  im_bboxes_blur = cv.addWeighted(im_bboxes_blur,0.2,im_black,0.8,0)  

  im_canny_optim = imx.gray2bgr(im_canny_optim)

  for x1,y1,x2,y2 in bboxes:
    cv.rectangle(im_canny_optim,(x1,y1),(x2,y2),(0,255,0),2)
    
  # draw logo on blurred frame
  for x1,y1,x2,y2 in bboxes_merged:
    im_bboxes_blur[y1:y2,x1:x2] = im_bboxes[y1:y2,x1:x2]
    cv.rectangle(im_bboxes_blur,(x1,y1),(x2,y2),(0,255,0),2)

  #cv.rectangle(im_canny_optim,(no_logo_c[:2]),(no_logo_c[2:4]),(0,0,120),1)
  cv.rectangle(im_canny_optim,(no_logo_rv[:2]),(no_logo_rv[2:4]),(0,0,120),1)
  cv.rectangle(im_canny_optim,(no_logo_rh[:2]),(no_logo_rh[2:4]),(0,0,120),1)
  
  #for r in quads:
  #  cv.rectangle(im_canny_optim,(r['region'][:2]),(r['region'][2:4]),(0,255,255),1)  

  r1 = np.hstack([imx.gray2bgr(im_canny_mean),imx.gray2bgr(im_canny_thresh)])
  r2 = np.hstack([im_canny_optim,im_bboxes_blur])
  im_comp = np.vstack([r1,r2])
  # save output
  if args.save_debug:
    # create composite
    
    fpath = join(args.output,'debug','{}.png'.format(name))
    cv.imwrite(fpath,im_comp)

  if args.display:
    cv.imshow(bname,im_comp)
    while True:
      k = cv.waitKey(0) & 0xFF
      if k == 27:
        break

  if not args.no_save_log:
    # write to log file
    bboxes_merged_str = '|'.join(['{},{},{},{},{}'.format(
      bbox[0],bbox[1],bbox[2],bbox[3],best_idx) for bbox,best_idx in zip(bboxes_merged,best_idxs)])
    txt = '{} | {} | {} |  {} | {} | {}\n'.format(
      bname,int(fps),nframes, ndupes, len(bboxes_merged), bboxes_merged_str)
    f_logfile = join(args.output,'log','log.csv')
    with open(f_logfile, "a") as fp:
      fp.write(txt)


  print('')

if __name__ == '__main__':

  vdir = join(cfg.DATA_STORE,'datasets/syrian_archive/video_snapshot_20171115/videos')
  vcat = 'barrel-bomb'
  vname ='99aef9afa4873b5b1ab43b0852debae32aea4f6246b11d1c90cad591d9dd393c.mp4'

  vcat = 'fab500-shn'
  vname = 'cfbffc06a0986f2b228c80c77cd5edc28d0682fe912fb63d6e91f09352303bfc.mp4'


  vdir = join(cfg.DATA_STORE,'datasets/syrian_archive/video_snapshot_20171115/')
  vcat = 'videos/barrel-bomb'
  vname = '5e39c9d6e7de23c056059bb93fb4930f3ea559e3da08bd0b7257b4ef3fe1537c.mp4'
  
  default_input = join(vdir,vcat,vname)
  default_output = os.path.join(vdir,'logos')
  
  import argparse
  ap = argparse.ArgumentParser()

  ap.add_argument('-i','--input',default=default_input,
    help="Path to input video, can be directory")
  ap.add_argument('-o','--output',default=default_output,
    help="Path to output directory where to save logos")

  # binary threshlding to remove non-stable edge features
  ap.add_argument('--bin_thresh_min',default=100,type=int,
    help="Binary threshold value for edge images")
  ap.add_argument('--bin_thresh_inc',default=5,type=int,
    help="Binary threshold increment per cycle")
  ap.add_argument('--bin_thresh_max',default=255,type=int,
    help="Binary threshold max value")
  ap.add_argument('--bin_var_thresh',default=800,type=float,
    help="Maximum tolerated variance of binary image")
  ap.add_argument('--canny_min',default=100,type=int,
    help="Canny min threshold") # 
  ap.add_argument('--canny_max',default=200,type=int,
    help="Canny min threshold") #150
  ap.add_argument('--canny_size',default=5,type=int,
    help="Canny kernel size (3, 5, or 7")
  ap.add_argument('--phash_thresh',default=6,type=int,
    help="Perceptual hash threshold: 1 (min) - 32 (max)")
  ap.add_argument('--min_num_samples',default=5,type=int,
    help="Minimum number of sample frames to analyze")
  ap.add_argument('--image_resolution',default=960,
    help="Image resolution for input to edge detection")
  ap.add_argument('--max_logo_size',default=(.35,.35),
    help="Max MSER size for logo in percentage of image (width,height)")
  ap.add_argument('--min_logo_size',default=(.02,.02),
    help="Min MSER size for logo in percentage of image (width,height)")
  ap.add_argument('--min_final_area',default=.003,type=float,
    help="Min final logo pixel area in percentage of image (width,height)")
  ap.add_argument('--max_final_area',default=.085,type=float,
    help="Max final logo pixel area in percentage of image (width,height)")
  ap.add_argument('--expand',default=12,
    help="Pixels to expand rectangles when searching for overlaps")
  ap.add_argument('--blur_amt',default=21,
    help="Blur amount to reduce graphic content (higher = more blur)")
  ap.add_argument('--max_ratio_wh',default=8,type=float,
    help="Max logo ratio for both w:h")
  ap.add_argument('--max_ratio_hw',default=2,type=float,
    help="Max logo ratio for h:w")
  ap.add_argument('--no_save',action='store_true',
    help="Flag to not save results")
  ap.add_argument('--display',action='store_true',
    help="Flag to not display images")
  ap.add_argument('--save_debug',action='store_true',
    help="Flag to save errors")
  ap.add_argument('--no_save_log',action='store_true',
    help="Flag to save output to log file")
  ap.add_argument('--sample_size',default=3,type=float,
    help="Seconds of video to analyze to create mean of edges")

  main(ap.parse_args())