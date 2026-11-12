from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from data_uncover import X_test, X_train

model = Sequential()
print(X_test.shape, X_train.shape)

model.add(Conv2D(32,(3,3), activation="relu", input_shape=(32,32,3)))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64,(3,3), activation="relu"))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(128,(3,3), activation="relu"))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer="adam", loss="categorical_crossentropy",metrics=['accuracy'])
