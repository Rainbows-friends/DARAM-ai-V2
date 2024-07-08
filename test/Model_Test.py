import os
import numpy as np
import cv2
from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pandas as pd
from random import sample
from datetime import datetime

KNOWN_FACES_DIR = 'C:\\Faceon_Project\\DTFO_Taeeun\\known_faces'
NON_FACES_DIR = 'C:\\Faceon_Project\\DTFO_Taeeun\\non_faces'
MODEL_PATH = 'C:\\Faceon_Project\\DARAM-AI(Deepface)\\face_recognition_model.keras'
MODEL_TEST_SUITE_DIR = 'C:\\Faceon_Project\\DARAM-AI(Deepface)\\Modeltest_suit'

def load_images_from_folder(folder, label):
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

non_face_images, non_face_labels = load_images_from_folder(NON_FACES_DIR, len(registered_faces))

all_images = known_images + non_face_images
all_labels = known_labels + non_face_labels

all_images = np.array(all_images)
all_labels = to_categorical(np.array(all_labels), num_classes=len(registered_faces) + 1)

X_train, X_test, y_train, y_test = train_test_split(all_images, all_labels, test_size=0.2, random_state=42)

if not os.path.exists(MODEL_PATH):
    print(f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
    exit()

model = load_model(MODEL_PATH)

num_tests = 30
test_indices = []
unknown_count = 0
for i in range(len(X_test)):
    if len(test_indices) >= num_tests:
        break
    true_label = np.argmax(y_test[i])
    if true_label == len(registered_faces):
        if unknown_count < 5:
            test_indices.append(i)
            unknown_count += 1
    else:
        test_indices.append(i)

test_images = X_test[test_indices]
true_labels = np.argmax(y_test[test_indices], axis=1)

predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

results = []
for i in range(num_tests):
    true_label = label_mapping.get(true_labels[i], "Unknown")
    predicted_label = label_mapping.get(predicted_labels[i], "Unknown")
    confidence = np.max(predictions[i])
    results.append({
        'Image Index': test_indices[i],
        'True Label': true_label,
        'Predicted Label': predicted_label,
        'Confidence': confidence
    })

os.makedirs(MODEL_TEST_SUITE_DIR, exist_ok=True)

current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

csv_file_path = os.path.join(MODEL_TEST_SUITE_DIR, f'test_results_{current_datetime}.csv')

results_df = pd.DataFrame(results)

# Apply styles and save to CSV
header = {
    'Image Index': 'Image Index',
    'True Label': 'True Label',
    'Predicted Label': 'Predicted Label',
    'Confidence': 'Confidence'
}

styled_results_df = results_df.style.set_table_styles(
    [{'selector': 'thead th', 'props': [('font-size', '12pt'), ('text-align', 'right'), ('font-weight', 'bold')]}],
    overwrite=False
).set_properties(
    **{'text-align': 'right', 'width': '100px'}
).set_caption("Model Test Results")

results_df.to_csv(csv_file_path, index=False)
print(f"Test results saved to '{csv_file_path}'")