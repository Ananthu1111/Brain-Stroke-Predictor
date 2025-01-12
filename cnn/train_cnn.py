import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os

# Set paths for train, test, and validation directories
train_dir = r'G:\mini project\Brain_Stroke_CT-SCAN_image\Train'
test_dir = r'G:\mini project\Brain_Stroke_CT-SCAN_image\Test'
val_dir = r'G:\mini project\Brain_Stroke_CT-SCAN_image\Validation'

# Data Preprocessing and Augmentation
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.2, 
                                   height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, 
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

# Loading the data and creating batches
batch_size = 32
img_size = (128, 128) # Resize all images to 128x128

train_generator = train_datagen.flow_from_directory(train_dir, target_size=img_size, 
                                                    batch_size=batch_size, class_mode='binary')

test_generator = test_datagen.flow_from_directory(test_dir, target_size=img_size, 
                                                  batch_size=batch_size, class_mode='binary')

val_generator = val_datagen.flow_from_directory(val_dir, target_size=img_size, 
                                                batch_size=batch_size, class_mode='binary')

# Build the Custom CNN Model
model = Sequential()

# Add convolutional layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the feature maps and add dense layers
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5)) # Adding dropout to prevent overfitting
model.add(Dense(1, activation='sigmoid')) # Sigmoid for binary classification (Normal/Stroke)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping to stop training if validation accuracy doesn't improve
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model for 30 epochs
history = model.fit(train_generator, epochs=30, validation_data=val_generator, 
                    callbacks=[early_stopping])

# Plot training & validation accuracy and loss
def plot_history(history):
    plt.figure(figsize=(12, 4))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

plot_history(history)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc*100:.2f}%")

# Save the trained model
model.save('G:/mini project/cnn/cnn_model.h5')
print("Model saved successfully!")