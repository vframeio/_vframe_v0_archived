# coding: utf-8

# Import the required modules
import os, sys
from os.path import join
import cv2 as cv
import time
import numpy as np
import imutils
from PIL import Image
import imagehash
from glob import glob
from skimage import feature
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
#sys.path.append('/vframe/vframe/app/main')
sys.path.append('/vframe/tools/')
from utils import imx
from utils import fiox
from processors.feature_extraction import feature_extractor_pytorch

def main(args):
  
  if args.generate_file_list:
    if os.path.isdir(args.input):
      if args.littlefork:
        # get list of directories
        print('[+] globbing file directory littlefork style...')
        files = glob(join(args.input,'*'))
      else:
        # get list of video files
        files = glob(join(args.input,'*.mp4'))
    else:
      files = [args.input]

    nfiles = len(files)
    if nfiles == 0:
      print('[-] No files found in "{}"'.format(args.input))
      sys.exit()
    else:
      print('Found {} files'.format(nfiles))

    print('[+] sorting files')
    files.sort()
    with open(args.file_list,'w') as fp:
      fp.write('\n'.join(files))
    print('[+] Wrote file list with {} lines'.format(len(files)))
    print('[+] Exiting.')
    sys.exit()
  else:
    with open(args.file_list,'r') as fp:
      files = fp.read().splitlines()
      nfiles = len(files)
      print('[+] Read file list with {} lines'.format(nfiles))


  #for i,f in enumerate(files[args.start_index:args.end_index]):
  if args.end_index is None:
    end_index = nfiles
  else:
    end_index = min(nfiles,args.end_index)

  start_index = max(args.start_index,0)
  files = files[start_index:end_index]
  nfiles = len(files)

  print('Going to process {} video files'.format(end_index-args.start_index))
  for i in range(nfiles):
    fp = files[i]
    per = int(100*i/nfiles)
    if args.littlefork:
      # get the first mp4 video file in directory
      fp_ls = glob(join(fp,'*.mp4'))
      if len(fp_ls) > 0:
        fp = fp_ls[0]
      else:
        continue
    if os.path.isfile(fp):
      print('[+] {}% ({}/{})- Process {}'.format(per,i,nfiles,os.path.basename(fp)))
      convert_vid2sceneframes(args,fp)
    print('')


def convert_vid2sceneframes(args,fpath):

  bname = os.path.basename(fpath)
  name,ext = os.path.splitext(bname)
  if args.littlefork:
    save_name = os.path.basename(os.path.dirname(fpath))
  else:
    save_name = bname

  # reset frames, max read is about 200FPS
  fps, nframes, frameset = imx.vid2frames(fpath,width=args.width)
  if nframes == 0:
    print('fps: {}, nframes: {}, len frames: {}'.format(fps,nframes,len(frameset)))
    print('[-] ERROR could not read movie {}'.format(bname))
    return
  print('[+] fps: {}, frames: {}'.format(fps,nframes))

  # Create Focal Mask
  # add active zones: horizontal and vertical
  try:
    h,w = frameset[0].shape[:2]
  except:
    print('[-] Error. No frames available for {}. Skipping video.'.format(fpath))
  focal_zone_dim = (0.33, 0.33)
  x1 = int(( (.5-(focal_zone_dim[0]/2)) * w ))
  x2 = int(( (.5+(focal_zone_dim[0]/2)) * w ))
  y1 = int(( (.5-(focal_zone_dim[1]/2)) * h ))
  y2 = int(( (.5+(focal_zone_dim[1]/2)) * h ))
  focal_zone_col = (x1,0,x2,h)
  focal_zone_row = (0,y1,w,y2)

  # add active zone: center rect
  focal_zone_center_dim = (0.5, 0.5)
  x1 = int(( (.5-(focal_zone_center_dim[0]/2)) * w ))
  x2 = int(( (.5+(focal_zone_center_dim[0]/2)) * w ))
  y1 = int(( (.5-(focal_zone_center_dim[1]/2)) * h ))
  y2 = int(( (.5+(focal_zone_center_dim[1]/2)) * h ))
  focal_zone_center = (x1,y1,x2,y2)

  focal_regions = [focal_zone_col, focal_zone_row,focal_zone_center]

  im_focal_mask = np.zeros_like(imx.bgr2gray(frameset[0])).astype(np.bool)
  for x1,y1,x2,y2 in focal_regions:
    im_focal_mask[y1:y2,x1:x2] = 1

  num_total_pixels = w*h
  num_focal_pixels = len(np.where(im_focal_mask > 0)[0])
  focal_mask_per = num_focal_pixels/(w*h)


  # # Use Canny Edges to find empty frames

  # mask focal region of canny images
  ims_canny_orig = [cv.Canny(f,args.canny_min,args.canny_max) for f in frameset]
  ims_canny = [np.logical_and(im.astype(np.bool),im_focal_mask) for im in ims_canny_orig] # focal mask

  # Compute sum of edge information in focal zone in canny images
  sum_thresh_per = 1. # max percentage of non-normalized summed edge pixels
  sum_thresh_pixels = sum_thresh_per/100.*num_focal_pixels # num pixels

  ims_canny_sums = np.array([ (len(np.where(im > 0)[0])) for im in ims_canny])
  canny_max = np.max(ims_canny_sums)
  ims_canny_sums_norm = [v/canny_max for v in ims_canny_sums]
  ims_canny_flags = np.array([v > sum_thresh_pixels for v in ims_canny_sums])

  # Visualize the frames that will be ignored
  ims_canny_keep_idxs = np.where(ims_canny_flags==1)[0]
  ims_canny_ignore_idxs = np.where(ims_canny_flags==0)[0]

  print('[+] Keeping frames more than {} edge pixels ({}%)'.format(int(sum_thresh_per/100.*num_focal_pixels),sum_thresh_per))
  print("[+] Keep: {}, Ignore: {}".format(len(ims_canny_keep_idxs),len(ims_canny_ignore_idxs)))

  # Use Pixel Mean to find empty frames
  # use focal mask to average only focal pixels
  im_focal_mask_uint8 = im_focal_mask.astype(np.uint8)
  mean_adj = num_total_pixels/num_focal_pixels
  ims_mean = np.array([mean_adj*np.mean(imx.bgr2gray(cv.bitwise_and(im,im,mask=im_focal_mask_uint8))) for im in frameset])
  #ims_mean = np.array([mean_adj*imx.bgr2gray(cv.bitwise_and(im,im,mask=im_focal_mask_uint8)) for im in frameset])
  max_mean = np.max(ims_mean)
  ims_mean_norm = ims_mean/max_mean
  ims_mean_flags = np.array([v > args.mean_thresh for v in ims_mean]) # Keep 1-flag frames

  # # Combine Edge/Mean Vals to Filter Empty Frames
  # Mark frames as 1 to keep, 0 to ignore
  # use logical OR on inverted logic to keep, then de-invert to keep 1(keep), 0(skip) structure
  ims_flags_comb = np.invert(np.logical_or(np.invert(ims_canny_flags),np.invert(ims_mean_flags)))
  print('[+] Combined mean + edge frames to skip: {}'.format(len(np.where(ims_flags_comb == 0)[0])))

  # gather the frames
  # Visualize the frames that will be ignored
  ims_keep_idxs = np.where(ims_flags_comb==1)[0]
  ims_ignore_idxs = np.where(ims_flags_comb==0)[0]

  # Load frames
  frames_ignore = [frameset[i] for i in ims_ignore_idxs]
  frames_keep = [frameset[i] for i in ims_keep_idxs]

  # TODO, inspect src --> dst error
  print('[+] Computing feature vectors...(ignoring warnings)')
  vals_phash = [imx.compute_phash(f) for f in frameset]
  
  #vals_feats = imx.compute_img2vec_feats(frameset,vals_phash,phash_thresh=1)

  # Paramerterize
  fe = feature_extractor_pytorch.FeatureExtractor(cuda=True,net='alexnet')
  vals_feats = imx.compute_features(fe,frameset,vals_phash,phash_thresh=2)

  scenes = []
  scene_idxs = []
  start_at = 0
  end_at = len(frameset)
  last_scene_idx = 0

  for idx in range(start_at,end_at):
    if ims_flags_comb[idx] == False:
      continue

    feat_delta = imx.cosine_delta(vals_feats[idx],vals_feats[last_scene_idx])
    phash_delta = (vals_phash[idx] - vals_phash[last_scene_idx])
      
    if feat_delta > args.thresh_feat_a or phash_delta > args.thresh_phash_a:
      last_scene_idx = idx
      scenes.append(scene_idxs)
      scene_idxs = []
      
    scene_idxs.append(idx)

  scenes.append(scene_idxs)

  # reduce scenes by removing transition frames
  min_scene_frames = int(fps*args.min_scene_duration)
  scenes_tmp = scenes.copy()
  scenes = [s for s in scenes_tmp if len(s) > min_scene_frames]

  scene_rep_idxs = []
  for scene in scenes:
    scene_edge_sum = ims_canny_sums[scene[0]:]
    # ensure there are 
    best_idx = np.argmax(scene_edge_sum)+scene[0]
    scene_rep_idxs.append(best_idx)
      
  print("[+] Feature based scenes: {}, best: {}".format(len(scenes),len(scene_rep_idxs)))

  # Get final scene indic
  scene_deduped_idxs = imx.dedupe_idxs(scene_rep_idxs,
    phashes=vals_phash,
    feats=vals_feats,
    feat_thresh=args.thresh_feat_b,
    phash_thresh=args.thresh_phash_b)

  # if too few, decrease thresholding
  nscene_frames = len(scene_deduped_idxs)
  print('[+] {} scenes with default thresh'.format(nscene_frames))

  if nscene_frames > 0 and nscene_frames < args.min_frames:
    print('[+] dec thresh')
    scene_deduped_idxs = imx.dedupe_idxs(scene_rep_idxs,
      phashes=vals_phash,
      feats=vals_feats,
      feat_thresh=args.thresh_feat_b/2,
      phash_thresh=args.thresh_phash_b/2)
  # if too few, increase thresholding
  elif nscene_frames > args.max_frames:
    print('[+] too many frames, increasing threshold')
    scene_deduped_idxs = imx.dedupe_idxs(scene_rep_idxs,
      phashes=vals_phash,
      feats=vals_feats,
      feat_thresh=args.thresh_feat_b*1.25,
      phash_thresh=args.thresh_phash_b*1.25)


  # Create dense keyframes representation with KMeans clusters
  if len(scene_deduped_idxs) < args.max_dense_frames:
    scene_frames_dense_idxs = scene_deduped_idxs.copy()
  else:
    dense_max_frames = min(args.max_dense_frames,len(scene_deduped_idxs))
    n_pca_components = min(12,len(scene_deduped_idxs))

    vec_length = len(vals_feats[0])
    vec_mat = np.zeros((len(scene_deduped_idxs), vec_length))

    for i in range(len(scene_deduped_idxs)):
      idx = scene_deduped_idxs[i]
      vec = vals_feats[idx]
      vec_mat[i, :] = vec

    print('[+] using {} PCA components'.format(n_pca_components))

    reduced_data = PCA(n_components=n_pca_components).fit_transform(vec_mat)
    kmeans = KMeans(init='k-means++', n_clusters=dense_max_frames, n_init=10)
    kmeans.fit(reduced_data)
    preds = kmeans.predict(reduced_data)
    
    scene_frames_dense_idxs = []
    clusters_used = []
    
    for i in range(len(scene_deduped_idxs)):
      idx = scene_deduped_idxs[i]
      if not preds[i] in clusters_used:
        clusters_used.append(preds[i])
        scene_frames_dense_idxs.append(idx)

  # reload video
  cap = cv.VideoCapture(fpath)
  print('[+] {}'.format(fpath))

  # Create list of poster frames (resized)
  # set capture object to frame index - 1 (read next frame)
  scene_frames_poster = []
  for idx in scene_deduped_idxs:
    cap.set(cv.CAP_PROP_POS_FRAMES, idx-1)
    res, frame = cap.read()
    scene_frames_poster.append(imutils.resize(frame,width=args.poster_im_width))
  

  # Create list of full resolution frames
  scene_frames_full = []
  for idx in scene_deduped_idxs:
    cap.set(cv.CAP_PROP_POS_FRAMES, idx-1)
    res, frame = cap.read()
    scene_frames_full.append(frame)

  scene_frames_poster_dense = []
  for idx in scene_frames_dense_idxs:
    cap.set(cv.CAP_PROP_POS_FRAMES, idx-1)
    res, frame = cap.read()
    scene_frames_poster_dense.append(imutils.resize(frame,width=args.poster_im_width))

  scene_frames_dense = []
  for idx in scene_frames_dense_idxs:
    cap.set(cv.CAP_PROP_POS_FRAMES, idx-1)
    res, frame = cap.read()
    scene_frames_dense.append(frame)

  # pad with empty frames
  if len(scene_frames_poster_dense) > 0:
    while len(scene_frames_poster_dense) < args.max_dense_frames:
      scene_frames_poster_dense.append(np.zeros_like(scene_frames_poster_dense[0]))

  #scene_frames_all = [frameset[idx] for idx in scene_rep_idxs]

  n = len(scene_rep_idxs)-len(scene_deduped_idxs)
  print('[+] {} scene keyframes removed from de-duplication'.format(n))
  print('[+] {} final scene frames'.format(len(scene_deduped_idxs)))

  if len(scene_deduped_idxs) > 0:
    # save the montage of all keyframes
    im_summary = imx.ims2montage(scene_frames_poster,ncols=args.num_cols)
    #fname = os.path.splitext(os.path.basename(f))[0]
    dir_poster = join(args.output,'posters')
    fiox.ensure_dir(dir_poster)
    fp_im = join(dir_poster,'{}.jpg'.format(save_name))
    cv.imwrite(fp_im,im_summary)

    # save the montage of dense keyframes
    im_summary = imx.ims2montage(scene_frames_poster_dense,ncols=args.num_cols_dense)
    #fname = os.path.splitext(os.path.basename(f))[0]
    dir_poster_dense = join(args.output,'posters_dense')
    fiox.ensure_dir(dir_poster_dense)
    fp_im = join(dir_poster_dense,'{}.jpg'.format(save_name))
    cv.imwrite(fp_im,im_summary)

    # output all video scene keyframes
    dir_frames_png = join(args.output,'frames_png',save_name)
    dir_frames_jpg = join(args.output,'frames_jpg',save_name)
    fiox.ensure_dir(dir_frames_png)
    fiox.ensure_dir(dir_frames_jpg)
    for idx,im in zip(scene_deduped_idxs,scene_frames_full):
      fp_im_png = join(dir_frames_png,'{:06d}.png'.format(idx))
      fp_im_jpg = join(dir_frames_jpg,'{:06d}.jpg'.format(idx))
      cv.imwrite(fp_im_png,im)
      cv.imwrite(fp_im_jpg,im)

    # write all dense keyframes for web
    dir_dense_frames_jpg = join(args.output,'frames_dense_jpg',save_name)
    fiox.ensure_dir(dir_dense_frames_jpg)
    for idx,im in zip(scene_frames_dense_idxs,scene_frames_dense):
      fp_im_jpg = join(dir_dense_frames_jpg,'{:06d}.jpg'.format(idx))
      cv.imwrite(fp_im_jpg,im)


  else:
    print('[+] no scenes found')

if __name__ == '__main__':
  import argparse
  ap = argparse.ArgumentParser()

  ap.add_argument('-i','--input',required=True,
    help="Path to input video or directory")
  ap.add_argument('-o','--output',required=True,
    help="Path to output directory")
  ap.add_argument('--width',default=160,type=int,
    help="Width of video for analysis (160-320 ideal)")
  ap.add_argument('--min_frames',default=3,type=int,
    help="Min number of frames (unless 0 frames")
  ap.add_argument('--max_frames',default=40,type=int,
    help="Max number of frames")

  # Thresholds for scene recognition first pass
  ap.add_argument('--thresh_phash_a',default=26,type=int,
    help="Min perceptual hash scene diff")
  ap.add_argument('--thresh_feat_a',default=0.125,type=float,
    help="Min perceptual hash scene diff (0.15 resnet-18)")

  # Thresholds for scene recognition second pass (de-duplication)
  ap.add_argument('--thresh_phash_b',default=24,type=int,
    help="Min perceptual hash scene diff")
  ap.add_argument('--thresh_feat_b',default=0.1875,type=float,
    help="Min perceptual hash scene diff (0.2125 resnet-18)")

  ap.add_argument('--min_scene_duration',default=0.5,type=float,
    help="Min scene duration in seconds")

  ap.add_argument('--mean_thresh',default=10,type=float,
    help="Min pixel mean for image focal area")

  ap.add_argument('--canny_min',default=100,type=int,
    help="Min canny threshold")
  ap.add_argument('--canny_max',default=200,type=int,
    help="Max canny threshold")

  ap.add_argument('--num_cols',default=4,type=int,
    help="Number of columns for poster file")
  ap.add_argument('--num_cols_dense',default=3,type=int,
    help="Number of columns for poster file")

  ap.add_argument('--poster_im_width',default=640,type=int,
    help="Image width for poster montage")
  ap.add_argument('--max_dense_frames',default=15,type=int,
    help="Maximum number of dense summary frames to save")

  ap.add_argument('--start_index',default=0,type=int,
    help="Start at this file index")
  ap.add_argument('--end_index',default=None,type=int,
    help="End at this file index")
  ap.add_argument('--littlefork',action="store_true",
    help="Search for videos in LittleFork style directory")
  ap.add_argument('--generate_file_list',action="store_true",
    help="Generate the master file list of videos")
  ap.add_argument('--file_list',default=None,
    help="Filepath to the master list of files (filename.txt")

  main(ap.parse_args())
