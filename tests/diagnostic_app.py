import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("Test 1 — import...")
import cv2
import numpy as np
print("✓ cv2 et numpy OK")

print("Test 2 — TensorFlow...")
from keras.models import load_model
print("✓ TensorFlow OK")

print("Test 3 — modèle...")
# Le chemin pointe bien vers le nouveau dossier 'models'
model = load_model('models/model_best.h5') 
print("✓ Modèle chargé OK")

print("Test 4 — MediaPipe...")
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path='mediapipe_models/hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)
print("✓ MediaPipe HandLandmarker OK")

print("Test 5 — caméra...")
cap = cv2.VideoCapture(0)
print(f"✓ Caméra ouverte : {cap.isOpened()}")
cap.release()

print("\n=== Tout est OK — le pipeline est prêt ! ===")