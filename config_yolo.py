import os

#
# path and dataset parameter for eval.py
#

DATASET_PATH = '/Users/shigetomi/Desktop/dataset_fit_noNegative/dataset_shisa/'
test_file_name = 'test_orig.txt'
TEST_FILE_PATH = os.path.join(DATASET_PATH, test_file_name)
GT_INFO_FILE_NAME = 'subwindow_log.txt'

#
# YOLO parameter
#
CKPT_FILE = './saved_model/yolov3_detect_specific_shisas.ckpt'
FROZEN_MODEL = ''
DATA_FORMAT = 'NHWC' # or NCHW
TINY = False
CLASS_NAME = './detect_seesaa/shisas.names' # coco class name
IMAGE_SIZE = 416

GPU_MEMORY_FRACTION = 0.9

#
# test parameter
#

CONF_THRESHOLD = 0.1
IOU_THRESHOLD = 0.1


#
# outputs
#

OUTPUT_DIR = './outputs/yolo/'
OUTPUT_LOG_name = 'log.csv'
OUTPUT_LOG_PATH = os.path.join(OUTPUT_DIR, OUTPUT_LOG_name)


#
# for debug
#

INPUT_IMAGE_PATH = '/Users/shigetomi/Desktop/sd1_cb6e8e65740b360d6167158be2251fcb4f9b979b.jpg'
OUTPUT_PATH = '/Users/shigetomi/Desktop/yolo_output.jpg'