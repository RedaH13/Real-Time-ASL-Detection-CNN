import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
from keras.models import load_model
from collections import deque
import mediapipe as mp

# ── 1. INITIALISATION ──
print("Chargement du modèle CNN...")
model = load_model('models/model_best.h5')

GESTES   = ['hello', 'thanks', 'yes', 'no', 'please', 'help', 'water', 'eat', 'good']
IMG_SIZE = 224
SEUIL    = 0.85

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="mediapipe_models/hand_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)
detector = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
phrase = []
dernier_geste = ''
prediction_buffer = deque(maxlen=5)

print("✓ Caméra ouverte :", cap.isOpened())

# ── 2. BOUCLE PRINCIPALE ──
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    h, w = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = detector.detect_for_video(mp_image, int(cap.get(cv2.CAP_PROP_POS_MSEC)))

    geste_affiche = ''
    conf = 0.0

    if result.hand_landmarks:
        # A. DESSINER LE SQUELETTE (OpenCV Manuel)
        for hand_landmarks in result.hand_landmarks:
            points = []
            for lm in hand_landmarks:
                px, py = int(lm.x * w), int(lm.y * h)
                points.append((px, py))
                
            connexions = [
                (0,1), (1,2), (2,3), (3,4),         
                (0,5), (5,6), (6,7), (7,8),         
                (5,9), (9,10), (10,11), (11,12),    
                (9,13), (13,14), (14,15), (15,16),  
                (13,17), (0,17), (17,18), (18,19), (19,20)
            ]
            
            for start_idx, end_idx in connexions:
                cv2.line(frame, points[start_idx], points[end_idx], (0, 255, 0), 2)
            for point in points:
                cv2.circle(frame, point, 3, (255, 255, 255), -1)

        # B. CALCULER LE CARRÉ PARFAIT (ROI)
        x_coords = [lm.x for lm in result.hand_landmarks[0]]
        y_coords = [lm.y for lm in result.hand_landmarks[0]]
        x_min, x_max = int(min(x_coords)*w), int(max(x_coords)*w)
        y_min, y_max = int(min(y_coords)*h), int(max(y_coords)*h)

        box_width, box_height = x_max - x_min, y_max - y_min
        center_x, center_y = x_min + (box_width // 2), y_min + (box_height // 2)
        half_size = int(max(box_width, box_height) * 1.5) // 2

        new_x_min, new_x_max = max(0, center_x - half_size), min(w, center_x + half_size)
        new_y_min, new_y_max = max(0, center_y - half_size), min(h, center_y + half_size)

        cv2.rectangle(frame, (new_x_min, new_y_min), (new_x_max, new_y_max), (255, 0, 0), 2)

        # C. CROP & PREDICTION
        roi_rgb = frame_rgb[new_y_min:new_y_max, new_x_min:new_x_max]

        if roi_rgb.size > 0:
            roi_resized = cv2.resize(roi_rgb, (IMG_SIZE, IMG_SIZE))
            input_data = np.expand_dims(roi_resized.astype(np.float32) / 255.0, axis=0)

            pred = model.predict(input_data, verbose=0)[0]
            idx = int(np.argmax(pred))
            conf = float(pred[idx])

            if conf >= SEUIL:
                prediction_buffer.append(idx)
                if len(prediction_buffer) == 5 and len(set(prediction_buffer)) == 1:
                    geste_affiche = GESTES[idx]
                    if geste_affiche != dernier_geste:
                        phrase.append(geste_affiche)
                        dernier_geste = geste_affiche
                        print(f"Reconnu : {geste_affiche} ({conf*100:.1f}%)")
            else: prediction_buffer.clear()
    else: prediction_buffer.clear()

    # ── 3. UI ──
    cv2.rectangle(frame, (0,0), (w,75), (20,20,20), -1)
    if geste_affiche: cv2.putText(frame, f'{geste_affiche.upper()} {conf*100:.0f}%', (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,100), 3)
    else: cv2.putText(frame, 'En attente...', (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (150,150,150), 2)

    cv2.rectangle(frame, (0,h-60), (w,h), (20,20,20), -1)
    cv2.putText(frame, 'Phrase : ' + ' '.join(phrase[-6:]), (15,h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

    cv2.imshow('Traducteur LSF (CNN + MediaPipe)', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord('c'): phrase.clear(); dernier_geste = ''

cap.release()
cv2.destroyAllWindows()