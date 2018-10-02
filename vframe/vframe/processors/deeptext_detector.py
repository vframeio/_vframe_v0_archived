# -*- coding: utf-8 -*-
#!/usr/bin/python
import sys
import os
import cv2 as cv
import numpy as np
import os.path as osp


def detect(args):
    
    DIR_MODELS = osp.join(args.data_store,'apps/syrian_archive/models/')
    PATH_MODEL = osp.join(DIR_MODELS,'text/deepscenetext/TextBoxes_icdar13.caffemodel')
    PATH_PROTO = osp.join(DIR_MODELS,'text/deepscenetext/textbox.prototxt')

    img = cv.imread(args.input)
    textSpotter = cv.text.TextDetectorCNN_create(PATH_PROTO, PATH_MODEL)

    rects, outProbs = textSpotter.detect(img);
    vis = img.copy()
    thres = 0.2

    for r in range(np.shape(rects)[0]):
        if outProbs[r] > thres:
            rect = rects[r]
            cv.rectangle(vis, (rect[0],rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 0, 0), 2)

    cv.imshow("Text detection result", vis)
    cv.waitKey()

def main():
    import argparse
    ap = argparse.ArgumentParser()
    DEFAULT_DATA_STORE = '/data_store/'
    ap.add_argument('-i','--input',required=True)
    ap.add_argument('--data_store',default=DEFAULT_DATA_STORE)
    args = ap.parse_args()

    detect(args)
    

if __name__ == "__main__":
    main()