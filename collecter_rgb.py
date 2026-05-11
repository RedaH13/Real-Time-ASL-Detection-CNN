import os
import cv2
import time
import mediapipe as mp

# Configuration
DATA_PATH = 'data_rgb'
GESTES    = ['hello', 'thanks', 'yes', 'no', 'please', 'help', 'water', 'eat', 'good']
NB_SEQ    = 30
LEN_SEQ   = 30
IMG_SIZE  = 224

# Initialisation
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands      = mp_hands.Hands(min_detection_confidence=0.7)
cap        = cv2.VideoCapture(0)

print("Création des dossiers...")
for geste in GESTES:
    for sequence in range(NB_SEQ):
        os.makedirs(os.path.join(DATA_PATH, geste, str(sequence)), exist_ok=True)

print("\nAppuyez sur 'Q' pour quitter l'enregistrement.")

for geste in GESTES:
    input(f"\n---> Préparez-vous pour le geste : {geste.upper()}. Appuyez sur ENTRÉE pour commencer...")
    
    for sequence in range(NB_SEQ):
        print(f"Enregistrement {geste} - Seq {sequence}/{NB_SEQ}...")
        
        for frame_num in range(LEN_SEQ):
            ret, frame = cap.read()
            if not ret: break
            
            # Le Recadrage Mathématique (ROI)
            h, w = frame.shape[:2]
            taille_roi = 224
            y1 = int(h/2 - taille_roi/2)
            y2 = int(h/2 + taille_roi/2)
            x1 = int(w/2 - taille_roi/2)
            x2 = int(w/2 + taille_roi/2)
            
            # Sauvegarde de l'image propre (avant MediaPipe)
            roi = frame[y1:y2, x1:x2]
            save_path = os.path.join(DATA_PATH, geste, str(sequence), f"{frame_num}.jpg")
            cv2.imwrite(save_path, roi)
            
            # Affichage UI
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Affichage du squelette pour le retour visuel
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            cv2.putText(frame, f'Enregistrement : {geste} | Seq: {sequence}', (15,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Collecte RGB', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit()
        
        time.sleep(1) # Pause entre les séquences

cap.release()
cv2.destroyAllWindows()
print("\nCollecte terminée !")