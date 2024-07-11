import os
import json
import random
import threading
import tkinter as tk
from tkinter import ttk

import cv2
import keras
import numpy as np
from keras._tf_keras.keras import Sequential
from keras._tf_keras.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

KNOWN_FACES_DIR = 'C:\\DARAM-ai-V2\\knows_faces'
current_dir = ""

def load_images_from_folder(folder, label=None, sample_size=None):
    global current_dir
    current_dir = folder
    images = []
    labels = []
    file_list = os.listdir(folder)

    if sample_size is not None and len(file_list) > sample_size:
        file_list = random.sample(file_list, sample_size)

    for filename in file_list:
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (128, 128))
            images.append(img)
            if label is not None:
                labels.append(label)
    return images, labels

def create_training_window():
    window = tk.Tk()
    window.title("Training Progress")
    window.geometry("600x200")

    progress_var = tk.DoubleVar()
    progress_bar = ttk.Progressbar(window, length=500, mode='determinate', variable=progress_var)
    progress_bar.pack(pady=20)

    epoch_label = tk.Label(window, text="Epoch: 0")
    epoch_label.pack()
    loss_label = tk.Label(window, text="Loss: 0.0000, Accuracy: 0.0000")
    loss_label.pack()
    val_loss_label = tk.Label(window, text="Val Loss: 0.0000, Val Accuracy: 0.0000")
    val_loss_label.pack()
    current_dir_label = tk.Label(window, text="Current Directory: None")
    current_dir_label.pack()

    def update_progress(epoch, logs):
        progress_var.set((epoch + 1) / 20 * 100)
        epoch_label.config(text=f"Epoch: {epoch + 1}")
        loss_label.config(text=f"Loss: {logs['loss']:.4f}, Accuracy: {logs['accuracy']:.4f}")
        val_loss_label.config(text=f"Val Loss: {logs['val_loss']:.4f}, Val Accuracy: {logs['val_accuracy']:.4f}")
        current_dir_label.config(text=f"Current Directory: {current_dir}")

    def on_train_end(logs=None):
        window.destroy()

    callback = keras.callbacks.LambdaCallback(on_epoch_end=update_progress, on_train_end=on_train_end)

    thread = threading.Thread(target=window.mainloop)
    thread.start()

    return callback

all_images = []
all_labels = []
valid_classes = []
label_mapping = {}

# Load known faces
for label, face_dir in enumerate(os.listdir(KNOWN_FACES_DIR)):
    face_path = os.path.join(KNOWN_FACES_DIR, face_dir)
    if face_dir == 'Other' or len(os.listdir(face_path)) < 200:
        continue
    valid_classes.append(face_dir)
    label_mapping[len(valid_classes)-1] = face_dir
    images, labels = load_images_from_folder(face_path, len(valid_classes)-1)
    all_images.extend(images)
    all_labels.extend(labels)

num_classes = len(valid_classes)
all_images = np.array(all_images)
all_labels = to_categorical(np.array(all_labels), num_classes=num_classes)

X_train, X_test, y_train, y_test = train_test_split(all_images, all_labels, test_size=0.2, random_state=42)

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_face_recognition_model_no_other_non_faces.keras', save_best_only=True, monitor='val_loss')

training_callback = create_training_window()

model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=20, validation_data=(X_test, y_test),
          callbacks=[training_callback, early_stopping, model_checkpoint])

model.save('face_recognition_model_no_other_non_faces.keras')
print("모델 저장 완료: face_recognition_model_no_other_non_faces.keras")

# Save the label mapping to a JSON file
with open('label_mapping_no_other_non_faces.json', 'w') as f:
    json.dump(label_mapping, f)

# 모델 검증
predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(y_test, axis=1)

print("테스트 셋에서의 정확도: ", np.mean(predicted_labels == true_labels))
for i in range(10):
    true_label = true_labels[i]
    predicted_label = predicted_labels[i]
    print(f"실제 라벨: {true_label}, 예측 라벨: {predicted_label}")