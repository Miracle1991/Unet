import  numpy as np
from keras.preprocessing.image import ImageDataGenerator
import Configure as cfg
import os
from skimage.io import imread
from PIL import ImageFile
import rle as rle
ImageFile.LOAD_TRUNCATED_IMAGES = True


dg_args = dict(featurewise_center = False,
               samplewise_center = False,
               rotation_range = 45,
               width_shift_range = 0.1,
               height_shift_range = 0.1,
               shear_range = 0.01,
               zoom_range = [0.9, 1.25],
               horizontal_flip = True,
               vertical_flip = True,
               fill_mode = 'reflect',
               data_format = 'channels_last')

if cfg.AUGMENT_BRIGHTNESS:
    dg_args[' brightness_range'] = [0.5, 1.5]
image_gen = ImageDataGenerator(**dg_args)

if cfg.AUGMENT_BRIGHTNESS:
    dg_args.pop('brightness_range')
label_gen = ImageDataGenerator(**dg_args)

def make_image_gen(in_df, batch_size = cfg.BATCH_SIZE):
    all_batches = list(in_df.groupby('ImageId'))
    out_rgb = []
    out_mask = []
    while True:
        np.random.shuffle(all_batches)
        for c_img_id, c_masks in all_batches:
            rgb_path = os.path.join(cfg.train_image_dir, c_img_id)
            c_img = imread(rgb_path)
            c_mask = np.expand_dims(rle.masks_as_image(c_masks['EncodedPixels'].values), -1)
            if cfg.IMG_SCALING is not None:
                c_img = c_img[::cfg.IMG_SCALING[0], ::cfg.IMG_SCALING[1]]
                c_mask = c_mask[::cfg.IMG_SCALING[0], ::cfg.IMG_SCALING[1]]
            out_rgb += [c_img]
            out_mask += [c_mask]
            if len(out_rgb)>=batch_size:
                #每生成BATCH_SIZE个数据后，返回归一化的值
                #使用np.stack可以生成供keras使用的数据类型
                yield np.stack(out_rgb, 0)/255.0, np.stack(out_mask, 0)
                out_rgb, out_mask=[], []

def create_aug_gen(in_gen, seed = None):
    np.random.seed(seed if seed is not None else np.random.choice(range(9999)))
    for in_x, in_y in in_gen:
        seed = np.random.choice(range(9999))
        # keep the seeds syncronized otherwise the augmentation to the images is different from the masks
        g_x = image_gen.flow(255*in_x,
                             batch_size = in_x.shape[0],
                             seed = seed,
                             shuffle=True)
        g_y = label_gen.flow(in_y,
                             batch_size = in_x.shape[0],
                             seed = seed,
                             shuffle=True)

        yield next(g_x)/255.0, next(g_y)