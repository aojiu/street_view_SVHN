from scipy.io import loadmat
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# load training and test data
train_mat=loadmat("train_32x32.mat")
test_mat=loadmat("test_32x32.mat")

# check shape of the image data
all_train_images = np.moveaxis(train_mat["X"], -1, 0)
all_test_images = np.moveaxis(test_mat["X"], -1, 0)

all_train_labels = train_mat["y"].reshape((-1,))
all_test_labels = test_mat["y"].reshape((-1,))
print("all_train_images shape: {}".format(all_train_images.shape))
print("all_test_images shape: {}".format(all_test_images.shape))

print("all_train_labels shape: {}".format(all_train_labels.shape))
print("all_test_labels shape: {}".format(all_test_labels.shape))

# normalize images
all_train_images_norm=all_train_images/255.0
all_test_images_norm=all_test_images/255.0

# convert the labels into one hot encoding
lb = LabelBinarizer()
all_train_labels = lb.fit_transform(all_train_labels)
all_test_labels = lb.fit_transform(all_test_labels)

# split the training data into training and validation
X_train, X_val, y_train, y_val = train_test_split(all_train_images_norm, all_train_labels, test_size=0.3, random_state=22)

X_train=X_train.astype("float32") 
X_val=X_val.astype("float32")

print("number of training instances: {}".format(X_train.shape))
print("number of validation instances: {}".format(X_val.shape))

print("number of training labels: {}".format(y_train.shape))
print("number of validation labels: {}".format(y_val.shape))


######################### construct the Lenet-5  #########################
model_lenet5 = keras.Sequential(
    [
        layers.Conv2D(6,(5,5), strides=(1,1), activation="tanh", padding="valid"),
        layers.AveragePooling2D((2,2), strides=(2,2)),
        
        layers.Conv2D(16,(5,5), strides=(1,1), activation="tanh", padding="valid"),
        layers.AveragePooling2D((2,2), strides=(2,2)),
        

        layers.Flatten(),
        layers.Dense(120, activation="tanh"),
        layers.Dense(84, activation="tanh"),
        layers.Dense(10, activation="softmax")
    ]
)

callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)


# a test run before compile the model
optimizer = tf.keras.optimizers.Adam(lr=1e-3, amsgrad=True)
# compile the model
model_lenet5.compile(loss=keras.losses.CategoricalCrossentropy(), optimizer=optimizer, metrics=["accuracy"])
# train the model using images
history_model_lenet5 = model_lenet5.fit(
    X_train,
    y_train,
    batch_size=64,
    epochs=200,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(X_val, y_val),
    callbacks=[callback]
)
model_lenet5.save_weights('./checkpoints/model_lenet5_checkpoint')


######################### construct the Lenet-5 with extra dropping layers #########################
model_lenet5_reg = keras.Sequential(
    [
        layers.Conv2D(6,(5,5), strides=(1,1), activation="tanh", padding="valid"),
        layers.AveragePooling2D((2,2), strides=(2,2)),
        
        layers.Conv2D(16,(5,5), strides=(1,1), activation="tanh", padding="valid"),
        layers.AveragePooling2D((2,2), strides=(2,2)),
        
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(120, activation="tanh"),
        layers.Dropout(0.3),
        layers.Dense(84, activation="tanh"),
        layers.Dense(10, activation="softmax")
    ]
)

callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3)

# a test run before compile the model
# model_lenet5(all_train_images_norm[5:6,:,:,:])
optimizer = tf.keras.optimizers.Adam(lr=1e-3, amsgrad=True)
# compile the model
model_lenet5_reg.compile(loss=keras.losses.CategoricalCrossentropy(), optimizer=optimizer, metrics=["accuracy"])
# model_lenet5.compile(loss=keras.losses.CategoricalCrossentropy(), optimizer="Adagrad", metrics=["accuracy"])
# train the model using images
history_model_lenet5_reg = model_lenet5_reg.fit(
    X_train,
    y_train,
    batch_size=64,
    epochs=200,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(X_val, y_val),
    callbacks=[callback]
)
model_lenet5_reg.save_weights('./checkpoints/model_lenet5_reg_checkpoint')




######################### construct the modified Lenet-5 #########################
model_lenet5_modified = keras.Sequential(
    [
        layers.Conv2D(16,(3,3), strides=(1,1), activation="relu", padding="same",input_shape=(32,32,3)),
        layers.Conv2D(16,(3,3), strides=(1,1), activation="relu", padding="same"),
        layers.AveragePooling2D((2,2), strides=(2,2)),
        
        layers.Conv2D(32,(3,3), strides=(1,1), activation="relu", padding="same"),
        layers.Conv2D(32,(3,3), strides=(1,1), activation="relu", padding="same"),
        layers.MaxPooling2D((2,2), strides=(2,2)),
        
        layers.BatchNormalization(),
        
        layers.Conv2D(64,(3,3), strides=(1,1), activation="relu", padding="same"),
        layers.Conv2D(120,(3,3), strides=(1,1), activation="relu", padding="same"),
        layers.AveragePooling2D((2,2), strides=(2,2)),
        
        
        layers.Dropout(0.3),
        
        layers.Flatten(),
        layers.Dense(120, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(84, activation="relu"),
        layers.Dense(10, activation="softmax")
    ]
)

callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3)

# a test run before compile the model
# model_lenet5(all_train_images_norm[5:6,:,:,:])
optimizer = tf.keras.optimizers.Adam(lr=1e-3, amsgrad=True)
# compile the model
model_lenet5_modified.compile(loss=keras.losses.CategoricalCrossentropy(), optimizer=optimizer, metrics=["accuracy"])
# train the model using images
history = model_lenet5_modified.fit(
    X_train,
    y_train,
    batch_size=64,
    epochs=200,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(X_val, y_val),
    callbacks=[callback]
)
model_lenet5_modified.save_weights('./checkpoints/model_lenet5_modified_checkpoint')

# inference on model_lenet5_modified
test_loss, test_acc = model_lenet5_modified.evaluate(x=all_test_images_norm, y=all_test_labels, verbose=0)
print("The test loss is: {}. The test accuracy is: {}".format(test_loss, test_acc))