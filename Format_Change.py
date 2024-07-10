import os
import numpy as np
import tensorflow as tf
from keras._tf_keras.keras.models import load_model, model_from_json

# 기존 모델 파일 경로
model_path = 'C:\\Faceon_Project\\DARAM-AI(Deepface)\\face_recognition_model_colab.keras'

# JSON과 가중치를 저장할 경로
json_path = 'C:\\Faceon_Project\\DARAM-AI(Deepface)\\model.json'
weights_path = 'C:\\Faceon_Project\\DARAM-AI(Deepface)\\model_weights.h5'

# 모델을 로드하고 JSON 파일로 저장
model = load_model(model_path)
model_json = model.to_json()

with open(json_path, 'w') as json_file:
    json_file.write(model_json)

model.save_weights(weights_path)

# JSON 파일로부터 모델을 다시 불러오기
with open(json_path, 'r') as json_file:
    loaded_model_json = json_file.read()

loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(weights_path)

# 새로운 경로에 다시 저장
new_model_path = 'C:\\Faceon_Project\\DARAM-AI(Deepface)\\face_recognition_model_keras.h5'
loaded_model.save(new_model_path)

print("모델이 성공적으로 저장되었습니다.")