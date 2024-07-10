import os
import random
import threading
import dlib
import cv2
import numpy as np
import keras
from keras import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if not tf.test.gpu_device_name():
    raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(tf.test.gpu_device_name()))

KNOWN_FACES_DIR = '/content/drive/MyDrive/colab/known_faces'
NON_FACES_DIR = '/content/drive/MyDrive/colab/non_faces'
OTHER_FACES_DIR = '/content/drive/MyDrive/colab/known_faces/Other'
PREDICTOR_PATH = '/content/drive/MyDrive/colab/shape_predictor_68_face_landmarks.dat'

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

BATCH_SIZE = 32

def get_face_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        coords = np.zeros((68, 2), dtype=int)
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords.flatten()
    return None

def process_image(filename, folder, label, use_landmarks, is_non_face):
    img_path = os.path.join(folder, filename)
    img = cv2.imread(img_path)
    if img is not None:
        if use_landmarks:
            if is_non_face:
                return np.zeros((136,), dtype=int), label
            else:
                landmarks = get_face_landmarks(img)
                if landmarks is not None:
                    return landmarks, label
        else:
            img = cv2.resize(img, (128, 128))
            return img, label
    return None, None

def load_images_from_folder(folder, label, use_landmarks=False, is_non_face=False, sample_size=None):
    images = []
    labels = []
    file_list = os.listdir(folder)
    if sample_size and sample_size < len(file_list):
        file_list = random.sample(file_list, sample_size)
    
    for filename in tqdm(file_list, desc=f"Loading {folder}"):
        img, lbl = process_image(filename, folder, label, use_landmarks, is_non_face)
        if img is not None:
            images.append(img)
            labels.append(lbl)

    return images, labels

def create_dataset(images, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((np.array(images), np.array(labels)))
    dataset = dataset.shuffle(buffer_size=len(images)).batch(batch_size)
    return dataset

def main():
    known_images = []
    known_labels = []
    label_mapping = {}
    registered_faces = [d for d in os.listdir(KNOWN_FACES_DIR) if d.isdigit()]

    for label, face_dir in enumerate(registered_faces):
        label_mapping[label] = face_dir
        face_path = os.path.join(KNOWN_FACES_DIR, face_dir)
        images, labels = load_images_from_folder(face_path, label, use_landmarks=True)
        known_images.extend(images)
        known_labels.extend(labels)

    non_face_images, non_face_labels = load_images_from_folder(NON_FACES_DIR, len(registered_faces), use_landmarks=True, is_non_face=True)
    other_face_images, other_face_labels = load_images_from_folder(OTHER_FACES_DIR, len(registered_faces), use_landmarks=True, sample_size=1000)

    all_images = known_images + non_face_images + other_face_images
    all_labels = known_labels + non_face_labels + other_face_labels

    all_labels = to_categorical(np.array(all_labels), num_classes=len(registered_faces) + 1)

    X_train, X_test, y_train, y_test = train_test_split(all_images, all_labels, test_size=0.2, random_state=42)

    train_dataset = create_dataset(X_train, y_train, BATCH_SIZE)
    test_dataset = create_dataset(X_test, y_test, BATCH_SIZE)

    with tf.device('/gpu:0'):
        model = Sequential([
            Dense(512, activation='relu', input_shape=(136,)),
            Dropout(0.5),
            BatchNormalization(),
            Dense(256, activation='relu'),
            Dropout(0.5),
            BatchNormalization(),
            Dense(len(registered_faces) + 1, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    class TrainingProgressCallback(keras.callbacks.Callback):
        def __init__(self):
            super().__init__()

        def on_epoch_end(self, epoch, logs=None):
            print(
                f"Epoch {epoch + 1}/{20} - loss: {logs['loss']:.4f} - accuracy: {logs['accuracy']:.4f} - val_loss: {logs['val_loss']:.4f} - val_accuracy: {logs['val_accuracy']:.4f}")

    if os.path.exists('/content/drive/MyDrive/colab/face_recognition_model.keras'):
        os.remove('/content/drive/MyDrive/colab/face_recognition_model.keras')

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('/content/drive/MyDrive/colab/best_face_recognition_model.keras',
                                       save_best_only=True, monitor='val_loss')

    callback = TrainingProgressCallback()

    with tf.device('/gpu:0'):
        model.fit(train_dataset, epochs=20, validation_data=test_dataset,
                  callbacks=[callback, early_stopping, model_checkpoint])

    model.save('/content/drive/MyDrive/colab/face_recognition_model.keras')
    print("모델 저장 완료: /content/drive/MyDrive/colab/face_recognition_model.keras")

    predictions = model.predict(test_dataset)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(y_test, axis=1)

    print("테스트 셋에서의 정확도: ", np.mean(predicted_labels == true_labels))
    for i in range(10):
        true_label = label_mapping.get(true_labels[i], "Unknown")
        predicted_label = label_mapping.get(predicted_labels[i], "Unknown")
        print(f"실제 라벨: {true_label}, 예측 라벨: {predicted_label}")

if __name__ == "__main__":
    main()
