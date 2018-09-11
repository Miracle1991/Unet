import os

# encoding: utf-8
BATCH_SIZE = 30
EDGE_CROP = 16
GAUSSIAN_NOISE = 0.1
UPSAMPLE_MODE = 'SIMPLE'
# downsampling inside the network
NET_SCALING = (1, 1)
# downsampling in preprocessing
IMG_SCALING = (4, 4)
# number of validation images to use
VALID_IMG_COUNT = 600
# maximum number of steps_per_epoch in training
MAX_TRAIN_STEPS = 1000
MAX_TRAIN_EPOCHS = 99
AUGMENT_BRIGHTNESS = False

'''
设置文件路径
'''
ship_dir = '/media/wdw/本地磁盘/深度学习数据集/airbus-kaggle/'
train_image_dir = os.path.join(ship_dir, 'train')
train_masks_dir = os.path.join('/home/wdw/DL/airbus/','train_ship_segmentations.csv')
test_image_dir = os.path.join(ship_dir, 'test')
sample_submission_path = '/home/wdw/DL/airbus/sample_submission.csv'