import cv2
import os
import numpy as np
from keras._tf_keras.keras.models import load_model
from deepface import DeepFace

KNOWN_FACES_DIR = 'C:\\DARAM-ai-V2\\knows_faces'
MODEL_PATH = 'C:\\DARAM-ai-V2\\face_recognition_model.keras'

label_mapping = {}
registered_faces = [d for d in os.listdir(KNOWN_FACES_DIR) if d.isdigit()]

for label, face_dir in enumerate(registered_faces):
    label_mapping[label] = face_dir

model = load_model(MODEL_PATH)

def detect_and_classify_faces(frame, model):
    try:
        detections = DeepFace.extract_faces(frame, detector_backend='mtcnn', enforce_detection=False)
        for face in detections:
            face_img = face['face']
            face_img_resized = cv2.resize(face_img, (128, 128))
            face_img_resized = np.expand_dims(face_img_resized, axis=0) / 255.0
            prediction = model.predict(face_img_resized)
            label = np.argmax(prediction)
            confidence = np.max(prediction)

            facial_area = face['facial_area']
            x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']

            if confidence > 0.5:
                name = label_mapping.get(label, "Unknown")
                if name != "Unknown":
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
                    cv2.putText(frame, f"{name} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
                else:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    except Exception as e:
        print(f"오류 발생: {str(e)}")

    return frame

def run_face_recognition():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # 좌우 반전
        frame = detect_and_classify_faces(frame, model)
        cv2.imshow('Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_face_recognition()
