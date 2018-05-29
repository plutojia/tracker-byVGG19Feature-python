import cv2
import sys
import time
import numpy as np

image_dir = 'test_data/BlurFace/BlurFace/img'
image_index = 1
groundtruth_rect_dir = 'test_data/BlurFace/BlurFace/groundtruth_rect.txt'

original_width = 576
original_height = 432


def load_NextImage(image_dir, image_index):
    image_path = "%s/%04d.jpg" % (image_dir, image_index)
    bgr_img = cv2.imread(image_path)
    return bgr_img


def changeImage(bgr_img):
    resized_img = cv2.resize(bgr_img, (224, 224))
    rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    rgb_img = rgb_img / 255.0
    return rgb_img


def readfile(filename):
    # return roi (1)
    bbox_list = [{}]  # padding index0 so that start iterater from 1
    with open(filename, 'r') as f:
        for line in f.readlines():
            linestr = line.strip()
            linestrlist = linestr.split()
            linelist = tuple([int(linestrlist[i]) for i in range(len(linestrlist))])
            bbox_list.append(linelist)
    return bbox_list


def draw_ROI(img, roi, color=(255, 0, 0)):
    img_height, img_width, channal = img.shape
    return cv2.rectangle(img, (int(img_width * roi["x1"]), int(img_height * roi["y1"])),
                         (int(img_width * roi["x2"]), int(img_height * roi["y2"])), color, 2)


def draw_bbox(img, bbox, color=(255, 0, 0)):
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    return cv2.rectangle(img, p1, p2, color, 2, 1)


def roi2bbox(roi, image_width, image_height):  # get bounding box x,y,w,h (>1)
    bbox = (int(image_width * roi["x1"]), int(image_height * roi["y1"]), int(image_width * (roi["x2"] - roi["x1"])),
            int(image_height * (roi["y2"] - roi["y1"])))
    return bbox


def bbox2roi(bbox, image_width, image_height):
    roi = {}
    roi["x1"] = bbox[0] / image_width
    roi["y1"] = bbox[1] / image_height
    roi["x2"] = (bbox[0] + bbox[2]) / image_width
    roi["y2"] = (bbox[1] + bbox[3]) / image_height
    return roi


def bbox_intersection(bbox1, bbox2):
    minx1 = bbox1[0]
    miny1 = bbox1[1]
    maxx1 = bbox1[0] + bbox1[2]
    maxy1 = bbox1[1] + bbox1[3]

    minx2 = bbox2[0]
    miny2 = bbox2[1]
    maxx2 = bbox2[0] + bbox2[2]
    maxy2 = bbox2[1] + bbox2[3]

    minx = max(minx1, minx2)
    miny = max(miny1, miny2)
    maxx = min(maxx1, maxx2)
    maxy = min(maxy1, maxy2)

    if minx > maxx:
        return 0
    if miny > maxy:
        return 0
    return (maxx - minx) * (maxy - miny)


def accuracy_overlapped_region(bbox1, bbox2):
    dim1 = bbox1[2] * bbox1[3]
    dim2 = bbox2[2] * bbox2[3]
    intersection = bbox_intersection(bbox1, bbox2)
    acc = intersection / (dim1 + dim2 - intersection)
    return acc


if __name__ == "__main__":
    '''
    label_list = readfile(groundtruth_rect_dir, original_width, original_height)
    cv2.namedWindow('bgr_img', flags=1)
    for i in range(500):
        bgr_img, rgb_img = load_NextImage(image_dir, image_index)
        roi = label_list[image_index]
        image_index += 1
        draw_ROI(bgr_img, roi)
        cv2.imshow("bgr_img", bgr_img)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27: break
    '''
    bbox1 = [2, 2, 5, 5]
    bbox2 = [0, 0, 3, 3]
    ins = bbox_intersection(bbox1, bbox2)
    acc = accuracy_overlapped_region(bbox1, bbox2)
    print(ins)
    print(acc)
