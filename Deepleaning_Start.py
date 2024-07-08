import os
import numpy as np
import cv2
from keras._tf_keras.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from keras._tf_keras.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf
import threading

KNOWN_FACES_DIR = 'C:\\Faceon_Project\\DTFO_Taeeun\\known_faces'
NON_FACES_DIR = 'C:\\Faceon_Project\\DTFO_Taeeun\\non_faces'
MODEL_PATH = 'face_recognition_model.h5'

if os.path.exists(MODEL_PATH):
    os.remove(MODEL_PATH)

def load_images_from_folder(folder, label):
    global current_directory, current_file_count
    current_directory = folder
    images = []
    labels = []
    for filename in os.listdir(folder):
        try:
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                img = cv2.resize(img, (128, 128))
                images.append(img)
                labels.append(label)
            current_file_count += 1
        except Exception as e:
            print(f"Error loading image {filename}: {str(e)}")
            continue
    return images, labels

known_images = []
known_labels = []
registered_faces = [d for d in os.listdir(KNOWN_FACES_DIR) if d.isdigit()]
total_files = sum([len(files) for r, d, files in os.walk(KNOWN_FACES_DIR)]) + sum(
    [len(files) for r, d, files in os.walk(NON_FACES_DIR)])

current_directory = None
current_file_count = 0

def display_loading_window():
    global current_directory, current_file_count, loading_window_open
    while current_file_count < total_files:
        img = np.zeros((300, 1000, 3), np.uint8)
        progress_text = f"Loading images: {current_file_count}/{total_files}"
        directory_text = f"Current directory: {current_directory if current_directory else 'None'}"
        cv2.putText(img, progress_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, directory_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.rectangle(img, (10, 150), (990, 200), (255, 255, 255), 2)
        progress_width = int((current_file_count / total_files) * 980)
        cv2.rectangle(img, (10, 150), (10 + progress_width, 200), (0, 255, 0), -1)
        cv2.imshow("Progress", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    loading_window_open = False
    cv2.destroyAllWindows()

loading_window_open = True
progress_thread = threading.Thread(target=display_loading_window)
progress_thread.start()
label_mapping = {}
for label, face_dir in enumerate(registered_faces):
    label_mapping[label] = face_dir
    face_path = os.path.join(KNOWN_FACES_DIR, face_dir)
    images, labels = load_images_from_folder(face_path, label)
    known_images.extend(images)
    known_labels.extend(labels)

non_face_images, non_face_labels = load_images_from_folder(NON_FACES_DIR, len(registered_faces))
all_images = known_images + non_face_images
all_labels = known_labels + non_face_labels
all_images = np.array(all_images)
all_labels = to_categorical(np.array(all_labels), num_classes=len(registered_faces) + 1)

X_train, X_test, y_train, y_test = train_test_split(all_images, all_labels, test_size=0.2, random_state=42)

model = Sequential([
    Input(shape=(128, 128, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(registered_faces) + 1, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

class TrainingProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self, total_steps):
        self.total_steps = total_steps
        self.current_step = 0
        self.current_epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch + 1

    def on_batch_end(self, batch, logs=None):
        self.current_step += 1
        self.display_training_window(logs)

    def on_epoch_end(self, epoch, logs=None):
        self.display_training_window(logs, is_epoch_end=True)

    def on_train_end(self, logs=None):
        self.display_training_window(logs, is_train_end=True)
        cv2.destroyAllWindows()
        model.save(MODEL_PATH)
        print("모델 저장 완료:", MODEL_PATH)
        os._exit(0)

    def display_training_window(self, logs, is_epoch_end=False, is_train_end=False):
        if not loading_window_open:
            img = np.zeros((600, 1000, 3), np.uint8)  # 크기를 늘림
            loading_text = f"Loading images: {current_file_count}/{total_files}"
            directory_text = f"Current directory: {current_directory if current_directory else 'None'}"
            cv2.putText(img, loading_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img, directory_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            loading_progress_width = int((current_file_count / total_files) * 980)
            cv2.rectangle(img, (10, 150), (10 + loading_progress_width, 200), (0, 255, 0), -1)
            training_text = f"Epoch: {self.current_epoch}, Step: {self.current_step}/{self.total_steps}"
            loss_text = f"Loss: {logs['loss']:.4f}, Accuracy: {logs['accuracy']:.4f}"
            val_loss = logs.get('val_loss', 0.0)
            val_accuracy = logs.get('val_accuracy', 0.0)
            val_loss_text = f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}"
            cv2.putText(img, training_text, (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img, loss_text, (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img, val_loss_text, (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.rectangle(img, (10, 400), (990, 450), (255, 255, 255), 2)
            training_progress_width = int((self.current_step / self.total_steps) * 980)
            cv2.rectangle(img, (10, 400), (10 + training_progress_width, 450), (0, 255, 0), -1)
            cv2.imshow("Progress", img)
            cv2.waitKey(1)
total_steps = (len(X_train) // 32) * 10
progress_callback = TrainingProgressCallback(total_steps)
cv2.destroyAllWindows()
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), callbacks=[progress_callback])