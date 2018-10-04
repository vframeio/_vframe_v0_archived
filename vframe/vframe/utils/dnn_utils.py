import cv2 as cv
import numpy as np

from vframe.models.metadata_item import DetectResult

# NMS post processing functions by @spmallick 
# https://github.com/spmallick/learnopencv/tree/master/ObjectDetection-YOLO
# https://www.learnopencv.com/deep-learning-based-object-detection-using-yolov3-with-opencv-python-c/


# Get the names of the output layers
def getOutputsNames(net):
  # Get the names of all the layers in the network
  layersNames = net.getLayerNames()
  # Get the names of the output layers, i.e. the layers with unconnected outputs
  return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def nms_cvdnn(net_outputs, dnn_threshold, nms_threshold):
  # Scan through all the bounding boxes output from the network and keep only the
  # ones with high confidence scores. Assign the box's class label as the class with the highest score.
  class_ids = []
  confidences = []
  boxes = []
  for net_output in net_outputs:
    for detection in net_output:
      scores = detection[5:]
      class_id = np.argmax(scores)
      confidence = scores[class_id]
      if confidence > dnn_threshold:
        cx = detection[0]
        cy = detection[1]
        w = detection[2]
        h = detection[3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = x1 + w
        y2 = y1 + h
        class_ids.append(class_id)
        confidences.append(float(confidence))
        boxes.append([x1, y1, x2, y2])

  # Perform non maximum suppression to eliminate redundant overlapping boxes with
  nms_objs = cv.dnn.NMSBoxes(boxes, confidences, dnn_threshold, nms_threshold)

  results = []
  for nms_obj in nms_objs:
    nms_idx = nms_obj[0]
    class_idx = class_ids[nms_idx]
    score = confidences[nms_idx]
    bbox = boxes[nms_idx]
    results.append(DetectResult(class_idx, score, bbox))

  return results