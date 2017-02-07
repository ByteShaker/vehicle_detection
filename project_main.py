import numpy as np
import cv2
import pickle

import matplotlib.pyplot as plt

from detection_functions.Vehicle_Classification import *
from detection_functions.Vehicle import *

from calibration.correctDistortion import correct_distortion
from detection_functions.sliding_window import *
from toolbox.draw_on_image import *
from detection_functions.detection_pipeline import *
from detection_functions.train_classifier import *

from scipy.ndimage.measurements import label

from moviepy.editor import VideoFileClip

import toolbox.multiple_image_out as mio
import toolbox.multiple_plots_out as mpo

MTX=None
DIST=None

VERBOSE=False

def process_image(raw_image, correct_distortion=False):
    img_shape = raw_image.shape

    if correct_distortion:
    # Correct Distortion with calculated Camera Calibration (If not present calibrate)
        global MTX, DIST, VERBOSE
        mtx, dist, process_image = correct_distortion(raw_image, mtx=MTX, dist=DIST, verbose=VERBOSE)
        if (MTX == None) | (DIST == None):
            MTX = mtx
            DIST = dist
    else:
        process_image = raw_image

    draw_image = np.copy(process_image)

    vehicle_collection.initalize_image(img_shape=process_image.shape, y_start_stop=[420, 720], xy_window=(360, 360), xy_overlap=(0.25, 0.9))
    vehicle_collection.find_hot_windows(process_image, vehicle_classification)

    #hot_window_frame_collection_conc = np.concatenate(hot_window_frame_collection)
    heatmap = np.zeros_like(process_image[:, :, 0]).astype(np.float)
    heatmap = add_heat(heatmap, vehicle_collection.hot_windows)

    if heatmap_frame_collection == None:
        heatmap_frame_collection = np.array(heatmap,ndmin=3)
    elif heatmap_frame_collection.shape[0] < 5:
        heatmap_frame_collection = np.append(heatmap_frame_collection,np.array(heatmap,ndmin=3), axis=0)
    else:
        heatmap_frame_collection = np.roll(heatmap_frame_collection, -1, axis=0)
        heatmap_frame_collection[-1,:] = np.array(heatmap,ndmin=2)

    heatmap = np.mean(heatmap_frame_collection,axis=0)

    heat_thresh = apply_threshold(heatmap,4)
    labels = label(heat_thresh)

    draw_image = draw_labeled_bboxes(draw_image,labels)

    #window_img = draw_boxes(draw_image, hot_windows_collection, color=(0, 0, 255), thick=6)

    #print(labels[1])
    #plt.imshow(labels[0], cmap='hot')
    #plt.show()
    cv2.imshow('Labels', draw_image)
    cv2.waitKey(1)

    #draw_image = cv2.cvtColor(draw_image, cv2.COLOR_BGR2RGB)

    return draw_image

if __name__ == "__main__":
    VERBOSE = True
    LEARN_NEW_CLASSIFIER = False

    vehicle_classification = Vehicle_Classification()
    vehicle_classification.train_classifier(LEARN_NEW_CLASSIFIER)

    vehicle_collection = Vehicle_Collection()

    heatmap_frame_collection = None



    #image = cv2.imread('./test_images/test1.jpg')
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #processed_image = process_image(image)
    #processed_image = cv2.cvtColor(processed_image,cv2.COLOR_RGB2BGR)
    #cv2.imshow('Resulting Image', processed_image)

    #cv2.imwrite('../output_images/test2_applied_lane_lines.jpg', combo)

    video_output = './project_video_calc_1.mp4'
    #clip1 = VideoFileClip('./project_video.mp4')
    clip1 = VideoFileClip('./test_video.mp4')
    #clip1 = VideoFileClip('../harder_challenge_video.mp4')

    white_clip_1 = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
    white_clip_1.write_videofile(video_output, audio=False)


