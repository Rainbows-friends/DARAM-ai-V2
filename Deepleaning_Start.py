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

KNOWN_FACES_DIR = 'C:\\DARAM-ai-V2\\knows_faces'
NON_FACES_DIR = 'C:\\DARAM-ai-V2\\knows_faces'
current_dir = ""

def load_images_from_folder(folder, label):
    global current_dir
    current_dir = folder
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (128, 128))
            images.append(img)
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

known_images = []
known_labels = []
label_mapping = {}
registered_faces = [d for d in os.listdir(KNOWN_FACES_DIR) if d.isdigit()]

for label, face_dir in enumerate(registered_faces):
    label_mapping[label] = face_dir
    face_path = os.path.join(KNOWN_FACES_DIR, face_dir)
    images, labels = load_images_from_folder(face_path, label)
    known_images.extend(images)
    known_labels.extend(labels)

all_images = known_images
all_labels = known_labels

all_images = np.array(all_images)
all_labels = to_categorical(np.array(all_labels), num_classes=len(registered_faces))

X_train, X_test, y_train, y_test = train_test_split(all_images, all_labels, test_size=0.2, random_state=42)

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
    Dense(len(registered_faces), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

if os.path.exists('face_recognition_model.keras'):
    os.remove('face_recognition_model.keras')

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_face_recognition_model.keras', save_best_only=True, monitor='val_loss')

training_callback = create_training_window()

model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), callbacks=[training_callback, early_stopping, model_checkpoint])

model.save('face_recognition_model.keras')
print("모델 저장 완료: face_recognition_model.keras")

predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(y_test, axis=1)

print("테스트 셋에서의 정확도: ", np.mean(predicted_labels == true_labels))
for i in range(10):
    true_label = label_mapping.get(true_labels[i], "Unknown")
    predicted_label = label_mapping.get(predicted_labels[i], "Unknown")
    print(f"실제 라벨: {true_label}, 예측 라벨: {predicted_label}")