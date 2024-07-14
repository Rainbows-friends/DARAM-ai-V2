import os

import cv2
from deepface import DeepFace

BASE_DIR = 'C:\\DARAM-ai-V2'
KNOWN_FACES_DIR = os.path.join(BASE_DIR, 'knows_faces')
LOG_FILE_PATH = os.path.join(BASE_DIR, 'log.txt')


def detect_faces(frame, log_file):
    detections = DeepFace.extract_faces(frame, detector_backend='mtcnn', enforce_detection=False)
    for face in detections:
        facial_area = face['facial_area']
        x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']

        if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
            continue

        color = (255, 255, 0)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        log_file.write(f"Detected face at ({x}, {y}, {w}, {h})\n")

    return frame


def run_face_detection():
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
            frame = detect_faces(frame, log_file)
            cv2.imshow('Face Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_face_detection()