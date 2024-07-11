import tensorflow as tf

model_path = "C://DARAM-ai-V2//face_recognition_model.keras"
model = tf.keras.models.load_model(model_path)

model.summary()