import os
import cv2
import numpy as np
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH  = 'data_rgb'
MODEL_PATH = 'models/model_best.h5'
GESTES     = ['hello', 'thanks', 'yes', 'no', 'please', 'help', 'water', 'eat', 'good']
IMG_SIZE   = 224

print("=== ÉVALUATION DU MODÈLE ===")
model = load_model(MODEL_PATH)

X_test = []
y_true = []

# On utilise seulement les 5 dernières séquences pour le test final
for label, geste in enumerate(GESTES):
    geste_path = os.path.join(DATA_PATH, geste)
    if not os.path.exists(geste_path): continue
        
    for seq in range(25, 30): 
        seq_path = os.path.join(geste_path, str(seq))
        if not os.path.exists(seq_path): continue
            
        for frame_num in range(30):
            img_path = os.path.join(seq_path, f"{frame_num}.jpg")
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img.astype(np.float32) / 255.0
                X_test.append(img)
                y_true.append(label)

X_test = np.array(X_test)
y_true = np.array(y_true)

print("Génération des prédictions...")
y_pred_probs = model.predict(X_test, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\n=== RAPPORT DE CLASSIFICATION ===")
print(classification_report(y_true, y_pred, target_names=GESTES))

print("Génération de la matrice de confusion...")
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=GESTES, yticklabels=GESTES)
plt.title('Matrice de Confusion CNN')
plt.ylabel('Vrai Geste')
plt.xlabel('Prédiction')
plt.tight_layout()

os.makedirs('metrics', exist_ok=True)
plt.savefig('metrics/matrice_confusion.png')
print("Matrice sauvegardée : metrics/matrice_confusion.png")