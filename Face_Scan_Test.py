import os
import cv2
import numpy as np
from deepface import DeepFace
from keras._tf_keras.keras.models import load_model

KNOWN_FACES_DIR = r'C:\DARAM-ai-V2\knows_faces'
MODEL_PATH_DET = r'C:\DARAM-ai-V2\test\face_detection_model_test.keras'
MODEL_PATH_REC = r'C:\DARAM-ai-V2\test\face_recognition_model_test.keras'
LOG_FILE_PATH = r'C:\DARAM-ai-V2\log.txt'

# 라벨 매핑 로드
label_mapping = {}
registered_faces = [d for d in os.listdir(KNOWN_FACES_DIR) if
                    os.path.isdir(os.path.join(KNOWN_FACES_DIR, d)) and d != 'Other']

for label, face_dir in enumerate(registered_faces):
    label_mapping[label] = face_dir

# 모델 로드
detection_model = load_model(MODEL_PATH_DET)
recognition_model = load_model(MODEL_PATH_REC)

def detect_and_classify_faces(frame, detection_model, recognition_model, log_file):
    detections = DeepFace.extract_faces(frame, detector_backend='mtcnn', enforce_detection=False)
    for face in detections:
        facial_area = face['facial_area']
        x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
        face_img = frame[y:y + h, x:x + w]
        face_img_resized = cv2.resize(face_img, (128, 128))
        face_img_resized = np.expand_dims(face_img_resized, axis=0) / 255.0

        detection_prediction = detection_model.predict(face_img_resized)
        if np.argmax(detection_prediction) == 1:  # 얼굴로 감지된 경우
            recognition_prediction = recognition_model.predict(face_img_resized)
            label = np.argmax(recognition_prediction)
            confidence = np.max(recognition_prediction)

            if confidence > 0.5:
                name = label_mapping.get(label, "Unknown")
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
                cv2.putText(frame, f"{name} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (255, 255, 0), 2)
            else:
                name = "Unknown"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

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
            frame = detect_and_classify_faces(frame, detection_model, recognition_model, log_file)
            cv2.imshow('Face Recognition', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_face_recognition()
