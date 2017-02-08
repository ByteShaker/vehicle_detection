from detection_functions.sliding_window import *
from detection_functions.detection_pipeline import *

class Vehicle():
    def __init__(self):
        self.vehicle_window = None


class Vehicle_Collection():
    def __init__(self):
        self.image_initialized = False
        self.img_shape = None
        self.precheck_windows = None
        self.hot_windows = None

        self.detected_vehicles = []

    def initalize_image(self, img_shape=(720,1280), y_start_stop=[436, 720], xy_window=(440, 440), xy_overlap = (0.5, 0.5)):
        self.image_initialized = True
        self.img_shape = img_shape
        self.precheck_windows = slide_precheck(img_shape, y_start_stop=y_start_stop, xy_window=xy_window, xy_overlap=xy_overlap)

    def find_hot_windows(self, process_image, vehicle_classification):
        self.hot_windows = search_windows(process_image, self.precheck_windows, vehicle_classification.classifier, vehicle_classification.X_scaler, color_space=vehicle_classification.color_space,
                                 spatial_size=vehicle_classification.spatial_size, hist_bins=vehicle_classification.hist_bins,
                                 orient=vehicle_classification.orient, pix_per_cell=vehicle_classification.pix_per_cell,
                                 cell_per_block=vehicle_classification.cell_per_block,
                                 hog_channel=vehicle_classification.hog_channel, spatial_feat=vehicle_classification.spatial_feat,
                                 hist_feat=vehicle_classification.hist_feat, hog_feat=vehicle_classification.hog_feat)