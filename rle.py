# encoding: utf-8

import numpy as np
from skimage.morphology import label
import os
import matplotlib.pyplot as plt
import pandas as pd

def multi_rle_encode(img, **kwargs):
    '''
    Encode connected regions as separated masks
    '''
    labels = label(img)
    if img.ndim > 2:
        return [rle_encode(np.sum(labels==k, axis=2), **kwargs) for k in np.unique(labels[labels>0])]
    else:
        return [rle_encode(labels==k, **kwargs) for k in np.unique(labels[labels>0])]

# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_encode(img, min_max_threshold=1e-3, max_mean_threshold=None):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    if np.max(img) < min_max_threshold:
        return '' ## no need to encode if it's all zeros
    if max_mean_threshold and np.mean(img) > max_mean_threshold:
        return '' ## ignore overfilled mask
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction

def masks_as_image(in_mask_list):
    # Take the individual ship masks and create a single mask array for all ships
    all_masks = np.zeros((768, 768), dtype = np.uint8)
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks |= rle_decode(mask)
    return all_masks

def masks_as_color(in_mask_list):
    # Take the individual ship masks and create a color mask array for each ships
    all_masks = np.zeros((768, 768), dtype = np.float)
    scale = lambda x: (len(in_mask_list)+x+1) / (len(in_mask_list)*2) ## scale the heatmap image to shift
    for i,mask in enumerate(in_mask_list):
        if isinstance(mask, str):
            all_masks[:,:] += scale(i) * rle_decode(mask)
    return all_masks

def pred_encode(img, **kwargs):
    cur_seg, _ = predict(img)
    cur_rles = multi_rle_encode(cur_seg, **kwargs)
    return [[img, rle] for rle in cur_rles if rle is not None]


if __name__ == '__main__':
    masks = pd.read_csv(os.path.join('/home/wdw/DL/airbus/', 'train_ship_segmentations.csv'))
    ship_dir = '/media/wdw/本地磁盘/深度学习数据集/airbus-kaggle/'
    train_image_dir = os.path.join(ship_dir, 'train')
    test_image_dir = os.path.join(ship_dir, 'test')

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize = (16, 5))

    rle_0 = masks.query('ImageId=="00021ddc3.jpg"')['EncodedPixels']
    img_0 = masks_as_image(rle_0)
    ax1.imshow(img_0)
    ax1.set_title('Mask as image')
    rle_1 = multi_rle_encode(img_0)
    img_1 = masks_as_image(rle_1)
    ax2.imshow(img_1)
    ax2.set_title('Re-encoded')
    img_c = masks_as_color(rle_0)
    ax3.imshow(img_c)
    ax3.set_title('Masks in colors')
    img_c = masks_as_color(rle_1)
    ax4.imshow(img_c)
    ax4.set_title('Re-encoded in colors')
    print('Check Decoding->Encoding',
          'RLE_0:', len(rle_0), '->',
          'RLE_1:', len(rle_1))
    print(np.sum(img_0 - img_1), 'error')
    plt.show()