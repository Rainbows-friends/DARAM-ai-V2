import os
from datetime import datetime
from random import sample
from tf_keras.models import load_model

import cv2
import numpy as np
import pandas as pd
from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

KNOWN_FACES_DIR = 'C:\\DARAM-ai-V2\\knows_faces'
MODEL_TEST_SUITE_DIR = 'C:\\DARAM-ai-V2\\Modeltest_suit'


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

all_images = known_images
all_labels = known_labels

all_images = np.array(all_images)
all_labels = to_categorical(np.array(all_labels), num_classes=len(registered_faces))

X_train, X_test, y_train, y_test = train_test_split(all_images, all_labels, test_size=0.2, random_state=42)

model_path = 'C:\\DARAM-ai-V2\\face_recognition_model.keras'
model = load_model(model_path)

num_tests = 30
test_indices = sample(range(len(X_test)), num_tests)
test_images = X_test[test_indices]
true_labels = np.argmax(y_test[test_indices], axis=1)

predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

results = []
for i in range(num_tests):
    true_label = label_mapping.get(true_labels[i], "Unknown")
    predicted_label = label_mapping.get(predicted_labels[i], "Unknown")
    confidence = np.max(predictions[i])
    results.append({'Image Index': test_indices[i], 'True Label': true_label, 'Predicted Label': predicted_label,
        'Confidence': confidence})

os.makedirs(MODEL_TEST_SUITE_DIR, exist_ok=True)
current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
xlsx_file_path = os.path.join(MODEL_TEST_SUITE_DIR, f'test_results_{current_datetime}.xlsx')

results_df = pd.DataFrame(results)

with pd.ExcelWriter(xlsx_file_path, engine='xlsxwriter') as writer:
    results_df.to_excel(writer, index=False, sheet_name='Test Results')
    workbook = writer.book
    worksheet = writer.sheets['Test Results']

    header_format = workbook.add_format(
        {'bold': True, 'text_wrap': True, 'valign': 'top', 'fg_color': '#D7E4BC', 'border': 1})

    for col_num, value in enumerate(results_df.columns.values):
        worksheet.write(0, col_num, value, header_format)

    for i, col in enumerate(results_df.columns):
        max_len = results_df[col].astype(str).map(len).max()
        worksheet.set_column(i, i, max_len + 2)

print(f"Test results saved to '{xlsx_file_path}'")
