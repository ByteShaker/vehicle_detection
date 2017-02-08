import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2

from detection_functions.feature_extraction import *
from toolbox.draw_on_image import *



# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # 1) Define an empty list to receive features
    img_features = []
    # 2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(img)
    # 3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # 4) Append features to list
        img_features.append(spatial_features)
    # 5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        # 6) Append features to list
        img_features.append(hist_features)
    # 7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:, :, channel],
                                                     orient, pix_per_cell, cell_per_block,
                                                     vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # 8) Append features to list
        img_features.append(hog_features)

    # 9) Return concatenated array of features
    #return np.concatenate(img_features)
    return img_features


# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB',
                   spatial_size=(32, 32), hist_bins=32,
                   hist_range=(0, 256), orient=9,
                   pix_per_cell=8, cell_per_block=2,
                   hog_channel=0, spatial_feat=True,
                   hist_feat=True, hog_feat=True):
    # 1) Create an empty list to receive positive detection windows
    on_windows = []
    current_window_area = ((0,0),(0,0))
    # 2) Iterate over all windows in the list
    for window in windows:
        # 3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        # 4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat,
                                       hist_feat=hist_feat, hog_feat=False)

        # 3.1) Detect Image area and perform Hog only once
        current_window_shape = ((window[1][1] - window[0][1]), (window[1][0] - window[0][0]))  # (delta_y,delta_x)

        if hog_feat:
            if (current_window_shape[0] != (current_window_area[1][1] - current_window_area[0][1])):
                if color_space != 'RGB':
                    if color_space == 'HSV':
                        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
                    elif color_space == 'LUV':
                        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
                    elif color_space == 'HLS':
                        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
                    elif color_space == 'YUV':
                        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
                    elif color_space == 'YCrCb':
                        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
                else:
                    feature_image = np.copy(img)
                current_window_area = ((0, window[0][1]), (feature_image.shape[1], window[1][1]))
                test_area = feature_image[current_window_area[0][1]:current_window_area[1][1],
                            current_window_area[0][0]:current_window_area[1][0]]
                scale = min(test_area.shape[0], test_area.shape[1]) / 64  # at most 64 rows and columns
                resized_test_area = cv2.resize(test_area,
                                               (np.int(test_area.shape[1] / scale), np.int(test_area.shape[0] / scale)))

                if hog_channel == 'ALL':
                    hog_features = []
                    for channel in range(resized_test_area.shape[2]):
                        hog_features.extend(get_hog_features(resized_test_area[:, :, channel],
                                                             orient, pix_per_cell, cell_per_block,
                                                             vis=False, feature_vec=False))
                else:
                    hog_features = get_hog_features(resized_test_area[:, :, hog_channel], orient,
                                                    pix_per_cell, cell_per_block, vis=False, feature_vec=False)

                # print(window)
            hog_feature_window_start = int((window[0][0] / scale) / pix_per_cell)
            extracted_hog_feature = np.array(hog_features)[:, hog_feature_window_start:hog_feature_window_start + 7]
            extracted_hog_feature = extracted_hog_feature.ravel()

            features.append(extracted_hog_feature)
        features = np.concatenate(features)
        # 5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        # 6) Predict using your classifier
        prediction = clf.predict(test_features)
        # 7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    # 8) Return windows for positive detections
    return on_windows


def add_heat(heatmap, bbox_list):
    bbox_list = np.array(bbox_list)

    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

