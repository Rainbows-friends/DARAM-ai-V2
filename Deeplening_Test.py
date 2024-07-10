import os
import keras
import numpy as np
import cv2
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras._tf_keras.keras.utils import to_categorical
from keras._tf_keras.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import ttk
import threading
import random

KNOWN_FACES_DIR = 'C:\\Faceon_Project\\DTFO_Taeeun\\known_faces'
OTHER_FACES_DIR = os.path.join(KNOWN_FACES_DIR, 'Other')
NON_FACES_DIR = 'C:\\Faceon_Project\\DTFO_Taeeun\\non_faces'
current_dir = ""


def load_images_from_folder(folder, label=None, sample_size=None):
    global current_dir
    current_dir = folder  # Update the current directory
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


# Load all images for face detection
all_images = []
all_labels = []

# Load all known faces including "Other"
for label, face_dir in enumerate(os.listdir(KNOWN_FACES_DIR)):
    face_path = os.path.join(KNOWN_FACES_DIR, face_dir)
    images, _ = load_images_from_folder(face_path)
    all_images.extend(images)
    all_labels.extend([1] * len(images))  # Label 1 for all faces

# Load non-faces
non_face_images, _ = load_images_from_folder(NON_FACES_DIR, sample_size=60)
all_images.extend(non_face_images)
all_labels.extend([0] * len(non_face_images))  # Label 0 for non-faces

# Prepare dataset for face detection (binary classification)
detection_images = np.array(all_images)
detection_labels = to_categorical(np.array(all_labels), num_classes=2)

X_train_det, X_test_det, y_train_det, y_test_det = train_test_split(detection_images, detection_labels, test_size=0.2,
                                                                    random_state=42)

# Build the detection model
detection_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')  # Binary classification
])

detection_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping_det = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint_det = ModelCheckpoint('best_face_detection_model_test.keras', save_best_only=True, monitor='val_loss')

training_callback_det = create_training_window()

detection_model.fit(X_train_det, y_train_det, epochs=20, validation_data=(X_test_det, y_test_det),
                    callbacks=[training_callback_det, early_stopping_det, model_checkpoint_det])

detection_model.save('face_detection_model_test.keras')
print("Detection 모델 저장 완료: face_detection_model_test.keras")

# Prepare dataset for face recognition (only registered faces)
registered_images = []
registered_labels = []
label_mapping = {}
registered_faces = [d for d in os.listdir(KNOWN_FACES_DIR) if
                    os.path.isdir(os.path.join(KNOWN_FACES_DIR, d)) and d != 'Other']

for label, face_dir in enumerate(registered_faces):
    label_mapping[label] = face_dir
    face_path = os.path.join(KNOWN_FACES_DIR, face_dir)
    images, labels = load_images_from_folder(face_path, label, sample_size=30)
    registered_images.extend(images)
    registered_labels.extend(labels)

# Prepare dataset for face recognition
recognition_images = np.array(registered_images)
recognition_labels = to_categorical(np.array(registered_labels), num_classes=len(registered_faces))

X_train_rec, X_test_rec, y_train_rec, y_test_rec = train_test_split(recognition_images, recognition_labels,
                                                                    test_size=0.2, random_state=42)

# Build the recognition model
recognition_model = Sequential([
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
    Dense(len(registered_faces), activation='softmax')  # Multi-class classification
])

recognition_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

if os.path.exists('face_recognition_model_test.keras'):
    os.remove('face_recognition_model_test.keras')

early_stopping_rec = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint_rec = ModelCheckpoint('best_face_recognition_model_test.keras', save_best_only=True,
                                       monitor='val_loss')

training_callback_rec = create_training_window()

recognition_model.fit(X_train_rec, y_train_rec, epochs=20, validation_data=(X_test_rec, y_test_rec),
                      callbacks=[training_callback_rec, early_stopping_rec, model_checkpoint_rec])

recognition_model.save('face_recognition_model_test.keras')
print("Recognition 모델 저장 완료: face_recognition_model_test.keras")

# 모델 검증
predictions = recognition_model.predict(X_test_rec)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(y_test_rec, axis=1)

print("테스트 셋에서의 정확도: ", np.mean(predicted_labels == true_labels))
for i in range(10):
    true_label = label_mapping.get(true_labels[i], "Unknown")
    predicted_label = label_mapping.get(predicted_labels[i], "Unknown")
    print(f"실제 라벨: {true_label}, 예측 라벨: {predicted_label}")
