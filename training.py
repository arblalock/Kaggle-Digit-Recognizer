# %%
#Import
import tensorflow as tf
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import np_utils
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers, regularizers
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.preprocessing.image import ImageDataGenerator
os.chdir('/workspaces/MachineLearning/Kaggle/Kaggle-Digit-Recognizer/')

# %%
#Data loading
train_data_raw = pd.read_csv('./data/train.csv')
submission_data = pd.read_csv('./data/test.csv')

#Globals
MODEL_SAVE_PATH = './saved_models/'
SUBM_SAVE_PATH = './submissions/'
TEST_SIZE = 0.1
RAND = 10
# features = ['keyword', 'location', 'text']
target = ['target']

train_data = train_data_raw.copy()
y = train_data.pop('label')


X_train, X_test, y_train, y_test = train_test_split(train_data, y, test_size=TEST_SIZE, random_state=RAND)
X_train_np = X_train.to_numpy()
X_test_np = X_test.to_numpy()

#convert to compatible shape for tensorflow
X_train_resize = np.resize(X_train_np, ( X_train_np.shape[0], 28,28,1))
X_test_resize  = np.resize(X_test_np , ( X_test_np.shape[0], 28,28,1))

#Make sure images are loading correctly
pil_img = tf.keras.preprocessing.image.array_to_img(X_train_resize[0])
pil_img.save('test.png')

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

encoder = LabelEncoder()
encoder.fit(y_train.to_numpy())
encoded_Y_train = encoder.transform(y_train.to_numpy())
dummy_y_train = tf.keras.utils.to_categorical(encoded_Y_train)

encoder.fit(y_test.to_numpy())
encoded_Y_test = encoder.transform(y_test.to_numpy())

dummy_y_test= tf.keras.utils.to_categorical(encoded_Y_test)

training_set = train_datagen.flow(X_train_resize, y= dummy_y_train)
test_set = test_datagen.flow(X_test_resize, y= dummy_y_test)

# Initialising the CNN
model = tf.keras.models.Sequential()

#Convolution
model.add(tf.keras.layers.Conv2D(filters=28, kernel_size=3, activation='relu', input_shape=[28, 28, 1]))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
model.add(tf.keras.layers.Conv2D(filters=28, kernel_size=3, activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

#Flattening
model.add(tf.keras.layers.Flatten())

# Full Connection
model.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Output Layer
model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

# %%
# Train Model
BATCH_SIZE = 512
EPOCS = 5
LEARNING_RATE = 0.001
SHUFFLE = False
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
history = model.fit(x = training_set, validation_data = test_set, epochs=EPOCS, shuffle=SHUFFLE)


# plot
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Evaluate
loss, accuracy = model.evaluate(test_set)
y_pred = model.predict_classes(test_set)s


# %%
# Save Model
MODEL_NAME = '7_18_submission'
model.save(MODEL_SAVE_PATH+MODEL_NAME)

# %%
# Load Model
MODEL_NAME = '7_18_submission'
model = tf.keras.models.load_model(MODEL_SAVE_PATH+MODEL_NAME)


# %%
# Get Predictions
predictions = model.predict_classes(submission_df)

output = pd.DataFrame({'id': submission_data.id, 'target': np.ravel(predictions)})
output.to_csv(SUBM_SAVE_PATH + MODEL_NAME + '.csv', index=False)