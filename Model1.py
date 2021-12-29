from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

path = 'C:/Users/alexn/Desktop/TR/Data_set'

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.1,
                                   zoom_range = 0.1,
                                   vertical_flip=True,
                                   horizontal_flip=True) 

test_datagen = ImageDataGenerator(rescale = 1./255)

valid_datagen = ImageDataGenerator(rescale = 1./255)

train =  train_datagen.flow_from_directory(directory = path+'/train',
                                                     target_size=(128, 128),
                                                     color_mode="rgb",
                                                     batch_size=32,
                                                     class_mode="categorical",
                                                     shuffle=True,
                                                     seed=42)
test = test_datagen.flow_from_directory(
    directory=path+'/test',
    target_size=(128, 128),
    color_mode="rgb",
    batch_size=1,
    class_mode="categorical",
    shuffle=False,
    seed=42)

valid = valid_datagen.flow_from_directory(directory = path+'/valid',
                                          target_size=(128, 128),
                                          color_mode="rgb",
                                          batch_size=32,
                                          class_mode="categorical",
                                          shuffle=True,
                                          seed=42)


model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3),activation='relu',
                 input_shape=(128,128,3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(13, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
model.fit(train, batch_size=128, epochs=12, validation_data=valid)

model.save('C:/Users/alexn/Desktop/TR/Model1')