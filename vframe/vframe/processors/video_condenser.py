# coding: utf-8
"""
Condense a video file into most representative keyframes
Creates folders for
  - posters_dense: the most representative [12] keyframes as single image
  - posters: an image grid of all keyframes
  - keyframes_training: contains fullsize PNG file
  - keyframes: are JPG
    - 160
    - 320
    - 640
    - 1280

Keyframes are selected based on:
- combined threshold of perceptual hash and feature vector
- the perceptual hash measures visual-verbatim similarity
- the feature vector measures content-relative similarity scores
- video is first pre-processed to eliminate empty/black frames
- and then with perceptual hash to speed up the feature vector extraction
- the results are grouped then filtered
- scenes that are too short are removed
- then the best frame within each scene is selected using quantity of canny/edge pixels
  - this should be replaced with BRISQUE
- then the indices are clustered into the top X frames for a dense summary

------------------------------------------------------------------------------

TODO

- reduce image export quality to 85%, compare to imagemagick

------------------------------------------------------------------------------

"""
import os, sys
from os.path import join
from pprint import pprint
import json
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
from pathlib import Path
sys.path.append('/vframe/tools/')
from utils import imx
from utils import fiox
from processors import feature_extractor_pytorch as fext_pytorch
from config import cli as cfg


class VideoCondenser:

  def __init__(self):
    pass


  def verify_mediainfo(self,fpath):
    """Read media info, check basic video requirements"""
    info = fiox.mediainfo(fpath)
    min_frames = 100
    min_duration = 5000 # milliseconds
    min_dim = (200,200)
    try:
      vobj = info['video']
    except:
      return False

    if vobj['duration'] < min_duration:
      print('[-] Video too short ({:.2f} seconds. Minimum: {} seconds.'.format(
        float(vobj['duration'])/1000,min_duration/1000))
      return False
    elif vobj['frame_count'] < 100:
      print('[-] Video too short ({} frames. Minimum: {} frames.'.format(
        float(vobj['frame_count']),min_frames))
      return False
    elif int(vobj['width']) < min_dim[0] or int(vobj['height']) < min_dim[1]:
      print('[-] Video dimensions too small ({}x{} pixels. Minimum: {}x{} frames.'.format(
        vobj['width'],vobj['height'],min_dim[0],min_dim[1]))
      return False
    else:
      return True


  def compute_velocity(vals):
    """returns np.array of difference v2-v1, with i0 = 0"""
    vels = np.zeros_like(vals)
    vels[0] = 0 # first frame has no velocity
    for i,v in enumerate(vals[1:]):
      vels[i] = abs(v - vals[i-1])
    return vels

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



  def create_focal_mask(self,width,height,cross_dim=(0.33,0.33),center_dim=(0.5,0.5)):
    """Create a binary mask to center focal weight
    param: width: width of image
    param: height: height of image
    param: cross_dim: the vertical and horizontal width of the focal area
    param: center_dim: the ratio of the center focal area
    returns: Numpy.ndaray bool of the binary focal mask
    """

    w,h = width,height
    x1 = int(( (.5-(cross_dim[0]/2)) * w ))
    x2 = int(( (.5+(cross_dim[0]/2)) * w ))
    y1 = int(( (.5-(cross_dim[1]/2)) * h ))
    y2 = int(( (.5+(cross_dim[1]/2)) * h ))
    focal_zone_col = (x1,0,x2,h)
    focal_zone_row = (0,y1,w,y2)

    # add active zone in middle
    x1 = int(( (.5-(center_dim[0]/2)) * w ))
    x2 = int(( (.5+(center_dim[0]/2)) * w ))
    y1 = int(( (.5-(center_dim[1]/2)) * h ))
    y2 = int(( (.5+(center_dim[1]/2)) * h ))
    focal_zone_center = (x1,y1,x2,y2)

    focal_regions = [focal_zone_col, focal_zone_row,focal_zone_center]

    im_focal_mask = np.zeros((h,w)).astype(np.bool)
    for x1,y1,x2,y2 in focal_regions:
      im_focal_mask[y1:y2,x1:x2] = 1
      
    return im_focal_mask


  def dedupe_idxs(self,idxs,feats,phashes,feat_thresh=.2125,phash_thresh=24):
    """De-duplicate a group of feature-vectors and perceptual hashes
    param: idxs: a list of indices referring to feats and phashes
    param: feats: the full list of feature-vectors
    param: phashes: the full list of peceptual hashes
    param: feat_thresh: feature-vector threshold (0.15-.025, 0.2125 good)
    param: phash_thresh: perceptual hash threshold (18-26 works OK, 25 good)
    returns: list de-duplicated indices
    """
    scene_deduped_idxs = []
    for i in idxs:
      feat = feats[i]
      phash = phashes[i]
      is_dupe = False
      for j in scene_deduped_idxs:
        if i == j:
          continue
        feat_delta = imx.cosine_delta(feat,feats[j])
        phash_delta = abs(phash-phashes[j])
        if feat_delta < feat_thresh and phash_delta < phash_thresh:
          is_dupe = True
          break
      if not is_dupe:
        scene_deduped_idxs.append(i)

    return scene_deduped_idxs


  def condense_video(self,fpath,kwargs):


    if not self.verify_mediainfo(fpath):
      print('[-] could not verify metadata')
      return {}

    # Open video file and read frames into list (about 200FPS)
    fps, nframes, frameset = imx.vid2frames(fpath,width=kwargs['width'])

    # ----------------------------------------------------------
    # Create Focal Mask
    # ignore areas where there are likely logos or less interesting visual info
    # ----------------------------------------------------------
    try:
      h,w = frameset[0].shape[:2]
    except:
      return {}
    num_total_pixels = w*h
    im_focal_mask = self.create_focal_mask(w,h)
    num_focal_pixels = len(np.where(im_focal_mask > 0)[0])

    # ----------------------------------------------------------
    # Use Canny Edges to find empty frames
    # ----------------------------------------------------------

    # mask focal region of canny images
    ims_canny_orig = [cv.Canny(f,kwargs['canny_min'],kwargs['canny_max']) for f in frameset]
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

    # ----------------------------------------------------------
    # Use Pixel Mean to find empty frames
    # ----------------------------------------------------------

    # use focal mask to average only focal pixels
    im_focal_mask_uint8 = im_focal_mask.astype(np.uint8)
    mean_adj = num_total_pixels/num_focal_pixels
    ims_mean = np.array([mean_adj*np.mean(imx.bgr2gray(cv.bitwise_and(im,im,mask=im_focal_mask_uint8))) for im in frameset])
    #ims_mean = np.array([mean_adj*imx.bgr2gray(cv.bitwise_and(im,im,mask=im_focal_mask_uint8)) for im in frameset])
    max_mean = np.max(ims_mean)
    ims_mean_norm = ims_mean/max_mean
    ims_mean_flags = np.array([v > kwargs['mean_thresh'] for v in ims_mean]) # Keep 1-flag frames

    # ----------------------------------------------------------
    # Combine pixel means and canny edge to filter "empty" frames
    # Mark frames as 1 to keep, 0 to ignore
    # use logical OR on inverted logic to keep, then de-invert to keep 1(keep), 0(skip) structure
    # ----------------------------------------------------------

    ims_flags_comb = np.invert(np.logical_or(np.invert(ims_canny_flags),np.invert(ims_mean_flags)))

    # gather the frames
    # Visualize the frames that will be ignored
    ims_keep_idxs = np.where(ims_flags_comb==1)[0]
    ims_ignore_idxs = np.where(ims_flags_comb==0)[0]

    # Load frames
    frames_ignore = [frameset[i] for i in ims_ignore_idxs]
    frames_keep = [frameset[i] for i in ims_keep_idxs]

    
    # ----------------------------------------------------------
    # Compute perceptual hash
    # and skip frames that are too smiliar
    # ----------------------------------------------------------

    vals_phash = [imx.compute_phash(f) for f in frameset]
    
    # Paramerterize PyTorch or TensorFlow/Keras
    fe = fext_pytorch.FeatureExtractor(cuda=kwargs['cuda'],net=kwargs['pytorch_net'])
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
        
      if feat_delta > kwargs['thresh_feat_a'] or phash_delta > kwargs['thresh_phash_a']:
        last_scene_idx = idx
        scenes.append(scene_idxs)
        scene_idxs = []
        
      scene_idxs.append(idx)

    scenes.append(scene_idxs)

    #print('[+] {} original scenes'.format(len(scenes)))
    # reduce scenes by removing transition frames
    min_scene_frames = int(fps*kwargs['min_scene_duration'])
    scenes_tmp = scenes.copy()
    scenes = [s for s in scenes_tmp if len(s) > min_scene_frames]
    print('[+] {} original scenes with min duration'.format(len(scenes)))

    # Edge case: if no scenes with min duration, take midpoint of video
    if len(scenes) == 0:
      scenes = [[0,nframes]]
      print('[+] Edge case: no good scenes. Use entire video.')
      print('\tHappens with short videos with lots of editing/graphics')

    # TODO replace with BRISQUE
    scene_rep_idxs = []
    for scene in scenes:
      scene_edge_sum = ims_canny_sums[scene[0]:scene[len(scene)-1]]
      # ensure there are 
      best_idx = np.argmax(scene_edge_sum)+scene[0]
      scene_rep_idxs.append(best_idx)
    
    #print("[+] Feature based scenes: {}".format(len(scenes)))

    # Get final scene indic
    scene_default_idxs = self.dedupe_idxs(scene_rep_idxs,
      phashes=vals_phash,
      feats=vals_feats,
      feat_thresh=kwargs['thresh_feat_b'],
      phash_thresh=kwargs['thresh_phash_b'])

    # if too few, decrease thresholding
    nscene_frames = len(scene_default_idxs)
    # print('[+] {} scenes with default thresh'.format(nscene_frames))

    thresh_factor = 1
    if nscene_frames > 0 and nscene_frames < kwargs['min_frames']:
      thresh_factor = .5
      scene_default_idxs = self.dedupe_idxs(scene_rep_idxs,
        phashes=vals_phash,
        feats=vals_feats,
        feat_thresh=kwargs['thresh_feat_b']*thresh_factor,
        phash_thresh=kwargs['thresh_phash_b']*thresh_factor)
    # if too few, increase thresholding
    elif nscene_frames > kwargs['max_frames']:
      thresh_factor = 1.25
      scene_default_idxs = self.dedupe_idxs(scene_rep_idxs,
        phashes=vals_phash,
        feats=vals_feats,
        feat_thresh=kwargs['thresh_feat_b']*thresh_factor,
        phash_thresh=kwargs['thresh_phash_b']*thresh_factor)

    # Create expanded set of indices
    thresh_factor *= kwargs['expand'] # decrease the threshold to get more keyframes
    scene_expanded_idxs = self.dedupe_idxs(scene_rep_idxs,
      phashes=vals_phash,
      feats=vals_feats,
      feat_thresh=kwargs['thresh_feat_b']*thresh_factor,
      phash_thresh=kwargs['thresh_phash_b']*thresh_factor)

    # Create dense keyframes representation with KMeans clusters
    if len(scene_default_idxs) < kwargs['max_dense_frames']:
      scene_frames_dense_idxs = scene_default_idxs.copy()
    else:
      dense_max_frames = min(kwargs['max_dense_frames'],len(scene_default_idxs))
      n_pca_components = min(12,len(scene_default_idxs))

      vec_length = len(vals_feats[0])
      vec_mat = np.zeros((len(scene_default_idxs), vec_length))

      for i in range(len(scene_default_idxs)):
        idx = scene_default_idxs[i]
        vec = vals_feats[idx]
        vec_mat[i, :] = vec

      reduced_data = PCA(n_components=n_pca_components).fit_transform(vec_mat)
      kmeans = KMeans(init='k-means++', n_clusters=dense_max_frames, n_init=10)
      kmeans.fit(reduced_data)
      preds = kmeans.predict(reduced_data)
      
      scene_frames_dense_idxs = []
      clusters_used = []
      
      for i in range(len(scene_default_idxs)):
        idx = scene_default_idxs[i]
        if not preds[i] in clusters_used:
          clusters_used.append(preds[i])
          scene_frames_dense_idxs.append(idx)

    
    print('[+] Default: {}, (Dense: {}, Expanded: {})'.format(\
      len(scene_default_idxs),len(scene_frames_dense_idxs),len(scene_expanded_idxs)))
    scene_idxs = {}
    scene_idxs['basic'] = [int(i) for i in scene_default_idxs]
    scene_idxs['dense'] = [int(i) for i in scene_frames_dense_idxs]
    scene_idxs['expanded'] = [int(i) for i in scene_expanded_idxs]
    return scene_idxs



  def extract_keyframes(self,fp_src_video,scenes,sha256,kwargs):
    """Write scene indices to disk"""

    # this should be multithreaded
    cap = cv.VideoCapture(fp_src_video)
    sha256_path = fiox.sha256_tree(sha256)
    fp_dst_keyframes = join(kwargs['output'],sha256_path,sha256)
    fiox.ensure_dir(fp_dst_keyframes)

    # load video
    cap = cv.VideoCapture(fp_src_video) # this should be multithreaded
    sha256_tree = fiox.sha256_tree(sha256)

    # provision vars
    montage_im_width = kwargs['montage_image_width']
    num_zeros = cfg.FRAME_NAME_ZERO_PADDING
    im_sizes = cfg.WEB_IMAGE_SIZES


    # ---------------------------------------------
    # Copy keyframes from video
    # ---------------------------------------------

    # expanded contains all frames
    frameset = {}
    for idx in scenes['expanded']:
      cap.set(cv.CAP_PROP_POS_FRAMES, idx-1)
      res, frame = cap.read()
      frames_expanded.append(frame)

    # pad dense summary to maintain consistent row x col layout
    if len(frames_poster_dense) > 0:
      while len(frames_poster_dense) < kwargs['max_dense_frames']:
        frames_poster_dense.append(np.zeros_like(frames_poster_dense[0]))

    #output_keyframes = join(kwargs['output'],'keyframes',save_name)
    #output_keyframes_training = join(kwargs['output'],'keyframes_training',save_name)
    dir_frames = join(kwargs['output'],'frames',sha256_path,sha256)
    dir_posters_default = join(kwargs['output'],'posters/default',sha256_path,sha256)
    dir_posters_dense = join(kwargs['output'],'posters/dense',sha256_path,sha256)
    dir_posters_expanded = join(kwargs['output'],'posters/expanded',sha256_path,sha256)

    # save all exapnded frames to frames at raw size
    for idx,frame in zip(scenes['expanded'],frames_expanded):
      fp_im = join(dir_frames,'{:07d}'.format(idx),'index.png')
      fiox.ensure_dir(fp_im,parent=True)
      cv.imwrite(fp_im,frame)


    if len(scenes['basic']) > 0:
      # save the montage of all keyframes
      im_summary = imx.montage(frames_poster,ncols=kwargs['num_cols'])
      dir_poster = join(kwargs['output'],'posters')
      fiox.ensure_dir(dir_poster)

      fp_im = join(dir_poster,'index.jpg')
      cv.imwrite(fp_im,im_summary)

      # save the montage of dense keyframes
      im_summary = imx.montage(frames_poster_dense,ncols=kwargs['num_cols_dense'])
      dir_poster_dense = join(kwargs['output'],'posters_dense')
      fiox.ensure_dir(dir_poster_dense)
      fp_im = join(dir_poster_dense,'index.jpg')
      cv.imwrite(fp_im,im_summary)
      
      # write full size local files for training
      fiox.ensure_dir(dir_frames)
      frameset[idx] = frame
  
    if kwargs['scene_type']:
      scene_types = [kwargs['scene_type']]
    else:
      scene_types = ['basic', 'expanded', 'dense']

    for scene_type in scene_types:
      frames = []

      # choose scenes to write
      scene_type = 'expanded' if kwargs['verified'] else 'basic'
      for idx,im in zip(scenes[scene_type],frames_full):
        fp_im = join(dir_frames,'{:06d}.png'.format(idx))
        cv.imwrite(fp_im,im)

        for size_dir,width in jpg_sizes.items():
          d = join(output_keyframes,size_dir)
          im_sized = imutils.resize(im,width=width)
          fp_im = join(output_keyframes,size_dir,'{:06d}.jpg'.format(idx))
          cv.imwrite(fp_im,im_sized)


      # load frames
      for idx in scenes[scene_type]:
        frames.append(frameset[idx])

      # create montages
      if kwargs['output_montages'] is not None:
        dp_montage = join(kwargs['output_montages'], scene_type, sha256_tree,sha256)
        Path(dp_montage).mkdir(parents=True, exist_ok=True)
        ncols, nrows = kwargs['montage_size']
        im_montage = imx.montage(frames, nrows=nrows, ncols=ncols, width=montage_im_width)
        fp_im = join(dp_montage, 'index.jpg')
        cv.imwrite(fp_im, im_montage)

      # path to web images JPG
      dp_web = join(kwargs['output_web'], sha256_tree, sha256)
      Path(dp_web).mkdir(parents=True,exist_ok=True)

      # path to training PNG frames
      if kwargs['output_training'] is not None:
        dp_train = join(kwargs['output_training'], sha256_tree, sha256)
        Path(dp_train).mkdir(parents=True,exist_ok=True)

      for idx,im in zip(scenes[scene_type], frames):
        idx_zpad = str(idx).zfill(num_zeros)

        # save training images (PNG)
        if kwargs['output_training'] is not None:
          dp = join(dp_train, idx_zpad)
          Path(dp).mkdir(parents=True, exist_ok=True)
          fp = join(dp_train, idx_zpad, 'index.png')
          cv.imwrite(fp, im)

        # generate directories
        for label, width in im_sizes.items():
          dp = join(dp_web, idx_zpad, label)
          Path(dp).mkdir(parents=True, exist_ok=True)

        # generate images
        for label, width in im_sizes.items():
          fp = join(dp_web, idx_zpad, label, 'index.jpg')
          im_resized = imutils.resize(im, width=width)
          cv.imwrite(fp, im_resized)

    return True