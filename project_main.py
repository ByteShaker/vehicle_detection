import numpy as np
import cv2
import pickle

import matplotlib.pyplot as plt

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

    # Uncomment the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    # image = image.astype(np.float32)/255

    ### TODO: Tweak these parameters and see how the results change.
    color_space = 'LUV'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 12  # HOG orientations
    pix_per_cell = 8  # HOG pixels per cell
    cell_per_block = 2  # HOG cells per block
    hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"
    spatial_size = (32, 32)  # Spatial binning dimensions
    hist_bins = 32  # Number of histogram bins
    spatial_feat = True  # Spatial features on or off
    hist_feat = True  # Histogram features on or off
    hog_feat = True  # HOG features on or off


    hot_windows_collection = None
    y_start_stop = [440, 720]  # Min and max in y to search in slide_window()
    xy_window = (360, 360)
    xy_overlap = (0.8, 0.75)
    y_position = y_start_stop[1]

    while (xy_window[0] > 40) & (y_position > y_start_stop[0]):
        # print(xy_window,y_start_stop)
        y_step = int(xy_window[1] * (1.0 - xy_overlap[1]))

        #print(y_position,xy_window,[y_position - xy_window[0], y_position])

        windows = slide_window(process_image, x_start_stop=[None, None], y_start_stop=[y_position - xy_window[0], y_position],
                               xy_window=xy_window, xy_overlap=xy_overlap)

        hot_windows = search_windows(process_image, windows, svc, X_scaler, color_space=color_space,
                                     spatial_size=spatial_size, hist_bins=hist_bins,
                                     orient=orient, pix_per_cell=pix_per_cell,
                                     cell_per_block=cell_per_block,
                                     hog_channel=hog_channel, spatial_feat=spatial_feat,
                                     hist_feat=hist_feat, hog_feat=True)

        #l, u, v = cv2.split(cv2.cvtColor(process_image, cv2.COLOR_RGB2LUV))
        #test = draw_boxes(v, hot_windows)
        #cv2.imshow('Windows', test)
        #cv2.waitKey(1)

        if hot_windows_collection == None:
            hot_windows_collection = hot_windows
        else:
            hot_windows_collection = hot_windows_collection + hot_windows

        y_position = y_position - y_step
        x_width = int(((1. - ((y_start_stop[1] - y_position) / 280)) * (360 - 40)) + 40)
        xy_window = (x_width, x_width)

    heatmap = np.zeros_like(process_image[:, :, 0]).astype(np.float)
    heatmap = add_heat(heatmap, hot_windows_collection)

    heat_thresh = apply_threshold(heatmap,3)
    labels = label(heat_thresh)

    draw_image = draw_labeled_bboxes(draw_image,labels)

    #window_img = draw_boxes(draw_image, hot_windows_collection, color=(0, 0, 255), thick=6)

    #print(labels[1])
    #plt.imshow(labels[0], cmap='hot')
    #plt.show()
    cv2.imshow('Labels', draw_image)
    cv2.waitKey(1)

    draw_image = cv2.cvtColor(draw_image, cv2.COLOR_BGR2RGB)

    return draw_image

if __name__ == "__main__":
    VERBOSE = True
    LEARN_NEW_CLASSIFIER = False

    svc, X_scaler = train_classifier(learn_new_classifier=LEARN_NEW_CLASSIFIER)

    #image = cv2.imread('./test_images/test1.jpg')
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #processed_image = process_image(image)
    #processed_image = cv2.cvtColor(processed_image,cv2.COLOR_RGB2BGR)
    #cv2.imshow('Resulting Image', processed_image)

    #cv2.imwrite('../output_images/test2_applied_lane_lines.jpg', combo)

    video_output = './project_video_calc.mp4'
    #clip1 = VideoFileClip('./project_video.mp4')
    clip1 = VideoFileClip('./test_video.mp4')
    #clip1 = VideoFileClip('../harder_challenge_video.mp4')

    white_clip_1 = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
    white_clip_1.write_videofile(video_output, audio=False)


