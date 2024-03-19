import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import sklearn
from sklearn.model_selection import train_test_split
import h5py

photon_data = r"C:\Users\Dell\Downloads\SinglePhotonPt50_IMGCROPS_n249k_RHv1.hdf5"
electron_data = r"C:\Users\Dell\Downloads\SingleElectronPt50_IMGCROPS_n249k_RHv1.hdf5"


try:
    with h5py.File(photon_data, 'r') as f:
        photon_keys = list(f.keys())  # Get list of keys in the file
        photon_data = f['X'][:]  # Access dataset with key 'X'
        photon_labels = f['y'][:]  # Access labels with key 'y'
except KeyError:
    print("Keys 'X' or 'y' not found in the HDF5 file.")
except IOError:
    print("Error accessing the HDF5 file.")

# Print available keys in the photon data file
print("Available keys in photon data file:", photon_keys)


try:
    with h5py.File(electron_data, 'r') as f:
        electron_keys = list(f.keys())  # Get list of keys in the file
        electron_data = f['X'][:]  # Access dataset with key 'X'
        electron_labels = f['y'][:]  # Access labels with key 'y'
except KeyError:
    print("Keys 'X' or 'y' not found in the HDF5 file.")
except IOError:
    print("Error accessing the HDF5 file.")

# Available keys in the photon data file
print("Available keys in electron data file:", electron_keys)


# Assuming labels for photons are 0 and for electrons are 1
photon_labels = np.zeros(len(photon_data))
electron_labels = np.ones(len(electron_data))

# Combining data and labels
X = np.concatenate((photon_data, electron_data), axis=0)
y = np.concatenate((photon_labels, electron_labels), axis=0)

# Split ds into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Split training set into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


# Defining data augmentation and preprocessing
data_augmentation = tf.keras.Sequential([
    layers.experimental.preprocessing.Rescaling(1./255),
    layers.experimental.preprocessing.RandomFlip("horizontal"),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
])

# Apply data augmentation and prepare datasets
X_train = data_augmentation(X_train)
X_val = data_augmentation(X_val)
X_test = data_augmentation(X_test)

# ResNet-15 model using Keras Functional API
def resnet_15(input_shape=(32, 32, 2), num_classes=2):
    inputs = tf.keras.Input(shape=input_shape)

    # Residual blocks
    x = layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
    x = layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Flatten and full connected layers
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs, name='resnet_15')
    return model

# Building the model
model = resnet_15()

# Adjusting learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, epochs= 1, validation_data=(X_val, y_val)) 

# Evaluate model on test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy}')

# Saving model weights
model.save_weights('resnet15_weights.h5')

# Predict labels for the entire dataset
predicted_labels = model.predict(X)

# Convert predicted probabilities to class labels
predicted_classes = np.argmax(predicted_labels, axis=1)

# Map class labels to particle types
class_mapping = {0: 'Photon', 1: 'Electron'}
predicted_particle_types = [class_mapping[label] for label in predicted_classes]

# Counting correct predictions
correct_predictions = sum(1 for true_label, predicted_label in zip(y, predicted_classes) if true_label == predicted_label)

# Calculating accuracy
accuracy = correct_predictions / len(y)

print(f'Dataset classification accuracy: {accuracy * 100:.2f}%')




** output **
**Available keys in photon data file: ['X', 'y']**
**Available keys in electron data file: ['X', 'y']**

**9960/9960 [==============================] - 2804s 281ms/step - loss: 0.6932 - accuracy: 0.4998 - val_loss: 0.6931 - val_accuracy: 0.5006**
**3113/3113 [==============================] - 169s 54ms/step - loss: 0.6932 - accuracy: 0.5000**
**Test Accuracy: 0.4999598264694214**
**15563/15563 [==============================] - 846s 54ms/step**
**Dataset classification accuracy: 50.00%  **
