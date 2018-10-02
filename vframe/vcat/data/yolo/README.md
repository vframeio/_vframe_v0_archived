# VFRAME: YOLOV3

Training summary for `cluster_munition_06`

- make this a table
- generated on Monday Sept 18 15:24
- 340 training annotations
- 80 validation annotations


## Training

- make shell scripts executeable: `chmod +x *.sh`
- start training: `bash run_train_init.sh`
- after 1.000 iterations use multi-gpu: `bash run_train_resume.sh`

## Monitor Loss

- can stop training once loss < 0.6

## Tips

from [@AlexeyAB](https://github.com/AlexeyAB/darknet/)

- set `random=1` in .cfg-file to increase precision by training on different resolutions ([source](https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L788))
- increase image resolution (eg `width=608`, `height=608`) or any multiple of 32 to increaes prevision
    - possible sizes: 416, 448, 480, 512, 544, 576, 608, 640, 672, 704
- ensure that all objects are labeled. unlabeled objects are scored negatively
- dataset should include objects with varying scales, resolution, lighting, angles, backgrounds and include about 2,000 different images for each class
- use negative samples (images that do not contain any of classes) to improve results. these are includced by adding empty .txt files. use as many negative as positive samples
- for training for small objects set `layers = -1, 11` instead of <https://github.com/AlexeyAB/darknet/blob/6390a5a2ab61a0bdf6f1a9a6b4a739c16b36e0d7/cfg/yolov3.cfg#L720> and set `stride=4` instead of <https://github.com/AlexeyAB/darknet/blob/6390a5a2ab61a0bdf6f1a9a6b4a739c16b36e0d7/cfg/yolov3.cfg#L717>
- If you train the model to distinguish Left and Right objects as separate classes (left/right hand, left/right-turn on road signs, ...) then for disabling flip data augmentation - add `flip=0` here: https://github.com/AlexeyAB/darknet/blob/3d2d0a7c98dbc8923d9ff705b81ff4f7940ea6ff/cfg/yolov3.cfg#L17
- General rule - your training dataset should include such a set of relative sizes of objects that you want to detect: 
    - `train_network_width * train_obj_width / train_image_width ~= detection_network_width * detection_obj_width / detection_image_width`
    - `train_network_height * train_obj_height / train_image_height ~= detection_network_height * detection_obj_height / detection_image_height`
* to speedup training (with decreasing detection accuracy) do Fine-Tuning instead of Transfer-Learning, set param `stopbackward=1` here: <https://github.com/AlexeyAB/darknet/blob/6d44529cf93211c319813c90e0c1adb34426abe5/cfg/yolov3.cfg#L548>


## Measure diversity
