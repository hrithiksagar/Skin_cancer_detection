from keras.layers import Dense, Flatten, Convolution3D, MaxPooling2D, Convolution2D
from keras import Sequential
from keras.preprocessing.image import ImageDataGenerator

classifier = Sequential()
classifier.add(Convolution2D(32,3,3, input_shape=(32,32,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Flatten())
classifier.add(Dense(64,activation='relu'))
classifier.add(Dense(4,init ='uniform',activation = 'softmax'))
classifier.compile(optimizer='adam',metrics=['accuracy'],loss='sparse_categorical_crossentropy')

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)
#start training
train_set = train_datagen.flow_from_directory(
        'C:/Users/hrith/PycharmProjects/SCC3/data/train',
        target_size=(32,32),
        batch_size=25,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'C:/Users/hrith/PycharmProjects/SCC3/data/test',
        target_size=(32, 32),
        batch_size=25,
        class_mode='binary')

classifier.fit_generator(
        train_set,
        steps_per_epoch=1000,
        epochs=20,
        validation_data=test_set,
        validation_steps=1000)

classifier.save('SCC.hdf5')
classifier.save_weights('myweights_SCC.h5')
print(train_set.class_indices)




