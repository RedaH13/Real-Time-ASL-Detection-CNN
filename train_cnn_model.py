import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

DATA_PATH = 'data_rgb'
GESTES    = ['hello', 'thanks', 'yes', 'no', 'please', 'help', 'water', 'eat', 'good']
IMG_SIZE  = 224

print("1. Création des dossiers...")
os.makedirs('models', exist_ok=True)
os.makedirs('metrics', exist_ok=True)

print("2. Chargement des images (Cela peut prendre 1 à 2 minutes)...")
X = []
y = []

for label, geste in enumerate(GESTES):
    geste_path = os.path.join(DATA_PATH, geste)
    if not os.path.exists(geste_path): continue
    
    for seq in range(30):
        seq_path = os.path.join(geste_path, str(seq))
        if not os.path.exists(seq_path): continue
        
        for frame in range(30):
            img_path = os.path.join(seq_path, f"{frame}.jpg")
            img = cv2.imread(img_path)
            if img is not None:
                # Conversion en RGB et normalisation (Optimisation Mémoire)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                X.append(img)
                y.append(label)

X = np.array(X, dtype=np.float16) / 255.0  # Normalisation cruciale
y = to_categorical(y).astype(np.float16)

print(f"✓ Images chargées : {X.shape}")

print("3. Séparation Train / Test...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("4. Création de l'architecture CNN...")
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5), # Prévient l'overfitting
    Dense(len(GESTES), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ModelCheckpoint('models/model_best.h5', monitor='val_accuracy', save_best_only=True)
]

print("5. Lancement de l'entraînement...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=15,
    batch_size=32,
    callbacks=callbacks
)

# Sauvegarde finale
model.save('models/model_final.h5')

print("6. Génération de la courbe d'apprentissage...")
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Précision (Accuracy)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Perte (Loss)')
plt.legend()

plt.savefig('metrics/courbes_apprentissage.png')
print("Entraînement terminé ! Modèle et courbes sauvegardés.")