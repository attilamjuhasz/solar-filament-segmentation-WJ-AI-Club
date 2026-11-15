from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input, MaxPooling2D, concatenate
from tensorflow.keras.models import Model


def conv_block(x, n_filters):
  x = Conv2D(n_filters, (3, 3), padding="same", activation="relu")(x)
  x = Conv2D(n_filters, (3, 3), padding="same", activation="relu")(x)
  return x


def encoder_blocks(x, n_filters):
  x = conv_block(x, n_filters)
  p = MaxPooling2D((2, 2))(x)
  return x, p  # <-- Added missing return statement


def decoder_block(x, skip_connection, n_filters):
  x = Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding="same")(x)
  x = concatenate([x, skip_connection])
  x = conv_block(x, n_filters)
  return x


# In creatingModel.py
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

  # Binary segmentation output (1 channel, sigmoid activation)
  outputs = Conv2D(1, (1, 1), activation="sigmoid")(u1)
  model = Model(inputs=[inputs], outputs=[outputs])
  return model


# Update img_channels to 1 and use binary_crossentropy
model = unet_model(n_classes=1, img_height=256, img_width=256, img_channels=1)
model.compile(
    loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
)

