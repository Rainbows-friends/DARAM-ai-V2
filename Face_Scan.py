import os
import json
import cv2
import numpy as np
from deepface import DeepFace
from keras._tf_keras.keras.models import load_model

KNOWN_FACES_DIR = 'C:\\DARAM-ai-V2\\knows_faces'
MODEL_PATH = 'C:\\DARAM-ai-V2\\face_recognition_model_no_other_non_faces.keras'
LOG_FILE_PATH = 'C:\\DARAM-ai-V2\\log.txt'

# Load label mapping
with open('label_mapping_no_other_non_faces.json', 'r') as f:
    label_mapping = json.load(f)

# 모델 로드
model = load_model(MODEL_PATH)


def detect_and_classify_faces(frame, model, log_file):
    detections = DeepFace.extract_faces(frame, detector_backend='mtcnn', enforce_detection=False)
    for face in detections:
        facial_area = face['facial_area']
        x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']

        # Ensure the detected face coordinates are within the frame bounds
        if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
            continue

        face_img = frame[y:y + h, x:x + w]
        if face_img.size == 0:  # Check if face_img is not empty
            continue

        face_img_resized = cv2.resize(face_img, (128, 128))
        face_img_resized = np.expand_dims(face_img_resized, axis=0) / 255.0

        recognition_prediction = model.predict(face_img_resized)
        label = np.argmax(recognition_prediction)
        confidence = np.max(recognition_prediction)

        name = label_mapping.get(str(label), "Unknown")
        if name != "Unknown" and confidence > 0.5:
            color = (255, 255, 0)
        else:
            name = "Unknown"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{name} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # 로그 파일에 기록
        log_file.write(f"Detected face at ({x}, {y}, {w}, {h}) - Predicted: {name}, Confidence: {confidence:.2f}\n")

    return frame


def run_face_recognition():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    with open(LOG_FILE_PATH, 'w') as log_file:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame = detect_and_classify_faces(frame, model, log_file)
            cv2.imshow('Face Recognition', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_face_recognition()