import cv2
import numpy as np
from keras._tf_keras.keras.models import load_model
from deepface import DeepFace
import concurrent.futures

# 모델 로드
model = load_model('face_recognition_model.h5')

# 얼굴 인식 함수
def detect_and_classify_faces(frame):
    detections = DeepFace.extract_faces(frame, detector_backend='mtcnn', enforce_detection=False)

    if detections:
        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_face, face) for face in detections]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())

        for (label, confidence, facial_area) in results:
            x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
            if confidence > 0.5:  # 신뢰도 임계값을 설정하여 등록된 얼굴로 간주
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, f"Person {label} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    else:
        print("얼굴을 감지하지 못했습니다.")

    return frame

def process_face(face):
    try:
        face_img = face['face']
        face_img_resized = cv2.resize(face_img, (128, 128))
        face_img_resized = np.expand_dims(face_img_resized, axis=0) / 255.0
        prediction = model.predict(face_img_resized)
        label = np.argmax(prediction)
        confidence = np.max(prediction)
        facial_area = face['facial_area']
        return (label, confidence, facial_area)
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        return ("Unknown", 0.0, {'x': 0, 'y': 0, 'w': 0, 'h': 0})

if __name__ == "__main__":
    # 웹캠 시작
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 얼굴 검출 및 인식
        frame = detect_and_classify_faces(frame)

        # 결과 이미지 표시
        cv2.imshow('Face Recognition', frame)

        # 'q' 버튼을 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 리소스 해제
    cap.release()
    cv2.destroyAllWindows()
