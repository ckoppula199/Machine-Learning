"""
A Convolutional Neural Network to determine if a picture is of a Cat or a Dog
"""

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy as np

"""
---------------Data Pre-processing---------------
"""
# Preprocessing on training set

# Image augmentation to prevent overfitting on the training set
train_data_gen = ImageDataGenerator(
    rescale=1./255, # Feature scaling
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

training_set = train_data_gen.flow_from_directory(
    '../Data/Convolutional_Neural_Networks/dataset/training_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

# Preprocessing on test set

test_data_gen = ImageDataGenerator(
    rescale=1./255
)

test_set = train_data_gen.flow_from_directory(
    '../Data/Convolutional_Neural_Networks/dataset/test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

"""
---------------Building the Convolutional Neural Network---------------
"""

# Initialising the CNN
cnn = tf.keras.models.Sequential()
# Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 3)))
# Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2))
# Second Convolution layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2))
# Flattening
cnn.add(tf.keras.layers.Flatten())
# Full Connection {hidden layer}
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
# Output layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

"""
---------------Training the Convolutional Neural Network---------------
"""
# Compile the CNN
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics='accuracy')
# Train CNN on training set and evaluate it on the test set
cnn.fit(x=training_set, validation_data=test_set, epochs=25)

"""
---------------Making a Prediction---------------
"""
test_img = image.load_img('../Data/Convolutional_Neural_Networks/dataset/single_prediction/cat_or_dog_2.jpg', target_size=(64, 64))
# Convert PIL format to numpy array
test_img = image.img_to_array(test_img)
# Create a batch
test_img = np.expand_dims(test_img, axis=0)
result = cnn.predict(test_img)
print(f"Class indices are the following:\n{training_set.class_indices}")
print(result[0][0])
if result[0][0] > 0.5:
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction)
