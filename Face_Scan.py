import cv2
import numpy as np
from keras._tf_keras.keras.models import load_model
from deepface import DeepFace
import threading
import tkinter as tk
from tkinter import scrolledtext

model = load_model('face_recognition_model.keras')

label_mapping = {
    0: '1302',
    1: '1401',
    2: '1402',
    3: '1404',
    4: '1405',
    5: '1409',
    6: '1415'
}


def detect_and_classify_faces(frame, model, log_widget):
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

            if confidence > 0.75:
                name = label_mapping.get(label, "Unknown")
                if name != "Unknown":
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
                    cv2.putText(frame, f"{name} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (255, 255, 0), 2)
                else:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                log_widget.insert(tk.END, f"{name} detected with confidence {confidence:.2f}\n")
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                log_widget.insert(tk.END, f"Unknown detected with confidence {confidence:.2f}\n")
            log_widget.see(tk.END)
    except Exception as e:
        print(f"오류 발생: {str(e)}")

    return frame


def run_face_recognition(log_widget):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # 좌우 반전
        frame = detect_and_classify_faces(frame, model, log_widget)
        cv2.imshow('Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def create_log_window():
    root = tk.Tk()
    root.title("Face Recognition Log")

    log_widget = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=50, height=20, font=("Arial", 12))
    log_widget.pack(expand=True, fill='both')

    return root, log_widget


if __name__ == "__main__":
    root, log_widget = create_log_window()

    recognition_thread = threading.Thread(target=run_face_recognition, args=(log_widget,))
    recognition_thread.start()

    root.mainloop()
