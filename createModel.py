from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input, concatenate, Conv2DTranspose
from tensorflow.keras.models import Model

def conv_block(x, n_filters):
    x = Conv2D(n_filters, (3,3), padding='same', activation='relu')(x)
    x = Conv2D(n_filters, (3,3), padding='same', activation='relu')(x)
    return x

def encoder_blocks(x, n_filters):
    x = conv_block(x,n_filters)
    p = MaxPooling2D((2,2))(x)
    return x,p

def decoder_block(x, p, n_filters):
    x = Conv2DTranspose(n_filters, (2,2), strides=(2,2), padding='same')(x)
    x = concatenate([x, p])
    x = conv_block(x,n_filters)
    return x

def unet_model(n_classes, img_height, img_width, img_channels):
    inputs = Input((img_height, img_width, img_channels))

    c1, p1 = encoder_blocks(inputs, n_filters=16)
    c2, p2 = encoder_blocks(p1, n_filters=32)
    c3, p3 = encoder_blocks(p2, n_filters=64)
    c4, p4 = encoder_blocks(p3, n_filters=128)

    bridge = conv_block(p4, n_filters=256)

    u4 = decoder_block(bridge, c4, n_filters=128)
    u3 = decoder_block(u4, c3, n_filters=64)
    u2 = decoder_block(u3, c2, n_filters=32)
    u1 = decoder_block(u2, c1, n_filters=16)

    outputs = Conv2D(1, (1,1), activation='sigmoid')(u1)
    model = Model(inputs=[inputs], outputs=[outputs])
    return model
IMG_HEIGHT, IMG_WIDTH = 256, 256

model = unet_model(n_classes=1, img_height=IMG_HEIGHT, img_width=IMG_WIDTH, img_channels=1)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
