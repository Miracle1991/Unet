

import os
import numpy as np
import pandas as pd
import Configure as cfg
from batch_generator import make_image_gen,create_aug_gen
import gc

from Network import model
from keras.optimizers import Adam
from loss import show_loss,IoU
from keras import models, layers
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from skimage.io import imread
from skimage.morphology import binary_opening, disk
from tqdm import tqdm_notebook
import rle


import tensorflow as tf
from keras import backend as K

K.clear_session()
tf.reset_default_graph()

gc.enable() # memory is tight
'''
读取训练数据
'''
masks = pd.read_csv(cfg.train_masks_dir)
#平衡空数据和非空数据,进行训练
not_empty = pd.notna(masks.EncodedPixels)
empty = pd.isna(masks.EncodedPixels)
print('total images:\t', masks.ImageId.nunique())
print('total masks:\t', not_empty.sum())
print('empty images:\t', (~not_empty).sum())
nan_masks = masks.loc[pd.isna(masks.EncodedPixels)]
nan_masks = nan_masks.sample(not_empty.sum())
print('random select the same length as total masks',len(nan_masks))

empty_for_train = nan_masks.sample(not_empty.sum())
notnan_masks = masks.loc[pd.notna(masks.EncodedPixels)]

masks = pd.concat([empty_for_train,notnan_masks], ignore_index=True)
print('concat masks...')
print('final the length of masks is :',len(masks))

#增加一列ship
masks['ships'] = masks['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0)
#将数据分为训练和验证两部分
train_df, valid_df = train_test_split(masks,
                                      test_size=0.3,
                                      stratify=masks['ships'])
#去掉masks的'ships'列
train_df.drop(['ships'], axis=1, inplace=True)
valid_df.drop(['ships'], axis=1, inplace=True)

train_gen = make_image_gen(train_df)
train_x, train_y = next(train_gen)

print('请确认训练数据的维度和值的范围:')
print('x', train_x.shape, train_x.min(), train_x.max())
print('y', train_y.shape, train_y.min(), train_y.max())

valid_x, valid_y = next(make_image_gen(valid_df, cfg.VALID_IMG_COUNT))
print('请确认验证数据的维度和值的范围:')
print(valid_x.shape, valid_y.shape)

cur_gen = create_aug_gen(train_gen)
t_x, t_y = next(cur_gen)

#垃圾回收
gc.collect()
del train_gen
del train_y
del train_x

#建立模型
seg_model = model(t_x.shape[1:])

'''
callback list
'''
# weight_path="{}_weights.best.hdf5".format('seg_model')
#
# checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min',
#                              save_weights_only=True, period=1)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                   patience=1, verbose=1, mode='min',
                                   min_delta=0.0001, cooldown=0, min_lr=1e-8)

early = EarlyStopping(monitor="val_loss", mode="min", verbose=2,
                      patience=10)  # probably needs to be more patient, but kaggle time is limited

callbacks_list = [ early, reduceLROnPlat]


seg_model.compile(optimizer=Adam(1e-3, decay=1e-6), loss=IoU, metrics=['binary_accuracy'])

step_count = min(cfg.MAX_TRAIN_STEPS, train_df.shape[0] // cfg.BATCH_SIZE)
aug_gen = create_aug_gen(make_image_gen(train_df))
print('开始fit()')

seg_model.fit_generator(aug_gen,
                        steps_per_epoch=step_count,
                        epochs=cfg.MAX_TRAIN_EPOCHS,
                        validation_data=(valid_x, valid_y),
                        callbacks=callbacks_list,
                        workers=1  # the generator is not very thread safe
                        )
    # return loss_history



print('结束fit()')
# if np.min([mh.history['val_loss'] for mh in loss_history]) < -0.2:
#     break

print('打印loss...')
# show_loss(loss_history)

print('训练已完成！,开始测试...')

# seg_model.load_weights(weight_path)
seg_model.save('seg_model.h5')

pred_y = seg_model.predict(valid_x)
print(pred_y.shape, pred_y.min(axis=0).max(), pred_y.max(axis=0).min(), pred_y.mean())

if cfg.IMG_SCALING is not None:
    fullres_model = models.Sequential()
    fullres_model.add(layers.AvgPool2D(cfg.IMG_SCALING, input_shape=(None, None, 3)))
    fullres_model.add(seg_model)
    fullres_model.add(layers.UpSampling2D(cfg.IMG_SCALING))
else:
    fullres_model = seg_model
fullres_model.save('fullres_model.h5')


def predict(img, path=cfg.test_image_dir):
    print('predict',img)
    c_img = imread(os.path.join(path, c_img_name))
    c_img = np.expand_dims(c_img, 0)/255.0
    cur_seg = fullres_model.predict(c_img)[0]
    cur_seg = binary_opening(cur_seg>0.99, np.expand_dims(disk(2), -1))
    return cur_seg, c_img

test_paths = np.array(os.listdir(cfg.test_image_dir))

out_pred_rows = []
i = 0
for c_img_name in tqdm_notebook(test_paths):
    i += 1
    print(i)
    out_pred_rows += rle.pred_encode(c_img_name, min_max_threshold=1.0)

    sub = pd.DataFrame(out_pred_rows)
sub.columns = ['ImageId', 'EncodedPixels']
sub = sub[sub.EncodedPixels.notnull()]

sub1 = pd.read_csv(cfg.sample_submission_path)
sub1 = pd.DataFrame(np.setdiff1d(sub1['ImageId'].unique(), sub['ImageId'].unique(), assume_unique=True),columns=['ImageId'])
sub1['EncodedPixels'] = None
print(len(sub1), len(sub))

sub = pd.concat([sub, sub1])
print(len(sub))
sub.to_csv('submission.csv', index=False)
sub.head()

print('done')