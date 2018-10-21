# VFRAME: Visual Forensics and Metadata Extraction

VFRAME is a computer vision framework for visual forensics and metadata extraction tool. It's designed for human rights technologists working with large datasets (millions) of vidoes and photos. 

VFRAME runs on a single workstation with NVIDIA GPU(s) and allows a single technical operator to convert large quantities of visual information into useful metadata that can be integrated with APIs or visual search engines. VFRAME is currently designed for technologists.

VFRAME includes a utlity called [VCAT](https://github.com/vframeio/vcat) to search the metadata and can execute visual queries on 10M keyframes in less than 0.3 seconds.

**VFRAME is under daily development. This page will be updated often during Oct-December**

![](docs/images/vframe_screenshot_04.10.2018_02.png)

Demo of cluster munition detector (in progress)

Features:
 
- fast video keyframe detection using CNN feature vectors
- modular commands for integrating with OpenCV DNN, PyTorch and other image processing frameworks
- integration with Sugarcube media collection system
- integration with VCAT metadata API, visual query, and annotation system

Several tasks are optimized

- convert video to representative keyframe < 15s
- face detection > 100 FPS

But others are still slow:

- metadata stored in serialized format (JSON/Pickle)
- EAST text detection < 2 FPS

The framework is being designed to allow scaling from several thousand videos to several million. A comple

## Getting started

- Create conda environment: `conda env create -f environment.yml`

## TODO

October 21 - 31

- add oriented text detection polygons for EAST
- add CRNN text recognition (eng)
- add tesseract 4.0 OCR (eng, ara)
- add ROI image extraction
- intregrate FAISS build scripts from VCAT

Nov 1 - 31

- add face embedding extraction
- add options for data export to CSV for Pandas analysis
- demos for Yolo darknet training workflow
- add scripts for negative mining
- add scripts for object tracking + low-confidence detector ROI extraction
- explore options for data augmentation on aerial dataset
- develop metrics for objection detection models
- add TF OD API project builder
- add instructions for freezing/exporting TF for OpenCV compatability
- update DNN modules for OpenCV 4 (pending release)
- update image feature extractor for PyTorch 1.0
- migrate/fix keyframe detection script, integrate with PyTorch 1.0

Dec 1 - 31

- migrate JSON/Pickle (slow) to local DB (sqlite, mongo, or LMDB, hdf5)
- add pose detection test
- improve README
- give demo examples
- create demo videos
- create demo notebooks


---------------------

- VFRAME and vframe.io Copyright (c) 2017-2018 Adam Harvey
- VFRAME Phase 1 support provided by (https://prototypefund.de)[PrototypeFund.de] (German Federal Ministry of Education and Research) 
