import cv2
import myIOfunc
import kcftracker
import sys
import utils
import numpy as np
import tensorflow as tf

image_dir = 'test_data/Jogging/img'
image_index = 1
groundtruth_rect_dir = 'test_data/Jogging/groundtruth_rect.1.txt'
KCF_ON=1

if __name__ == '__main__':
    tracker = kcftracker.KCFTracker(False)
    KCFtracker = cv2.TrackerKCF_create()
    cv2.namedWindow('Tracking', flags=1)

    # initial
    bgr_img = myIOfunc.load_NextImage(image_dir, image_index)
    label_list = myIOfunc.readfile(groundtruth_rect_dir)
    real_bbox = label_list[image_index]
    image_index += 1
    acc_sum=0
    acc_sum_kcf=0
    print("real_bbox:", real_bbox)

    # Initialize tracker with first frame and bounding box
    ok = tracker.init(real_bbox,bgr_img)
    if ok:
        print("init MY tracker:", ok)
    if KCF_ON:
        ok = KCFtracker.init(bgr_img, real_bbox)
        if ok:
            print("init KCF tracker:", ok)

    while True:
        # Start timer
        timer = cv2.getTickCount()

        # Read a new frame
        bgr_img = myIOfunc.load_NextImage(image_dir, image_index)
        real_bbox = label_list[image_index]
        image_index += 1

        # Update tracker
        bbox = tracker.update(bgr_img)

        # Draw bounding box
        myIOfunc.draw_bbox(bgr_img, bbox, (0, 255, 0))
        accuracy = myIOfunc.accuracy_overlapped_region(bbox, real_bbox)
        acc_sum += accuracy
        # Display accuracy on frame
        cv2.putText(bgr_img, "acc : " + str(float(accuracy)), (50, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        # Draw groundtruth_rect
        myIOfunc.draw_bbox(bgr_img, real_bbox)

        if KCF_ON:
            ok, KCFbbox = KCFtracker.update(bgr_img)
            if not ok :
                KCFaccuracy = 0
                cv2.putText(bgr_img, "KCFacc : LOST" , (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                            (50, 50, 170), 2)
            else:
                myIOfunc.draw_bbox(bgr_img, KCFbbox, (0, 0, 255))
                KCFaccuracy = myIOfunc.accuracy_overlapped_region(KCFbbox, real_bbox)
                cv2.putText(bgr_img, "KCFacc : " + str(float(KCFaccuracy)), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        (50, 50, 170), 2)
            acc_sum_kcf += KCFaccuracy

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

        # Display FPS on frame
        cv2.putText(bgr_img, "FPS : " + str(int(fps)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        # Display result
        cv2.imshow("Tracking", bgr_img)

        acc_average=acc_sum/image_index
        acc_average_kcf=acc_sum_kcf/image_index

        print("acc_average",acc_average,"      acc_average_kcf:",acc_average_kcf)
        # Exit if ESC pressed
        k = cv2.waitKey(50) & 0xff
        if k == 27: break
