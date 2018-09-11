"""  Build U-Net model """
from keras import models, layers
import Configure as cfg


def model(input_shape):
    input_img = layers.Input(input_shape, name='RGB_Input')
    print('input_img')

    if cfg.NET_SCALING is not None:
        pp_in_layer = layers.AvgPool2D(cfg.NET_SCALING)(input_img)

    pp_in_layer = layers.GaussianNoise(cfg.GAUSSIAN_NOISE)(pp_in_layer)
    pp_in_layer = layers.BatchNormalization()(pp_in_layer)

    conv1 = layers.Conv2D(16, 3, activation='relu', padding='same')(pp_in_layer)
    conv1 = layers.Conv2D(16, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = layers.Conv2D(32, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(32, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = layers.Conv2D(64, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool4)
    conv5 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv5)

    up6 = layers.Conv2D(128, 2, activation='relu', padding='same')(layers.UpSampling2D(size=(2, 2))(conv5))
    merge6 = layers.concatenate([conv4, up6])
    conv6 = layers.Conv2D(128, 3, activation='relu', padding='same')(merge6)
    conv6 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv6)

    up7 = layers.Conv2D(64, 2, activation='relu', padding='same')(layers.UpSampling2D(size=(2, 2))(conv6))
    merge7 = layers.concatenate([conv3, up7])
    conv7 = layers.Conv2D(64, 3, activation='relu', padding='same')(merge7)
    conv7 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv7)

    up8 = layers.Conv2D(32, 2, activation='relu', padding='same')(layers.UpSampling2D(size=(2, 2))(conv7))
    merge8 = layers.concatenate([conv2, up8])
    conv8 = layers.Conv2D(32, 3, activation='relu', padding='same')(merge8)
    conv8 = layers.Conv2D(32, 3, activation='relu', padding='same')(conv8)

    up9 = layers.Conv2D(16, 2, activation='relu', padding='same')(layers.UpSampling2D(size=(2, 2))(conv8))
    merge9 = layers.concatenate([conv1, up9])
    conv9 = layers.Conv2D(16, 3, activation='relu', padding='same')(merge9)
    conv9 = layers.Conv2D(16, 3, activation='relu', padding='same')(conv9)
    conv10 = layers.Conv2D(1, 1, activation='sigmoid')(conv9)

    up10 = layers.UpSampling2D(cfg.NET_SCALING)(conv10)

    return models.Model(inputs=[input_img], outputs=up10)