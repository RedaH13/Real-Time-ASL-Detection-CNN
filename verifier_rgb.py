import os
import cv2
import numpy as np

DATA_PATH = 'data_rgb'
GESTES    = ['hello', 'thanks', 'yes', 'no', 'please', 'help', 'water', 'eat', 'good']

print("=== VÉRIFICATION DU DATASET RGB ===")

total_images = 0
erreurs = 0

for geste in GESTES:
    chemin_geste = os.path.join(DATA_PATH, geste)
    if not os.path.exists(chemin_geste):
        print(f"Dossier manquant : {geste}")
        continue
        
    for seq in range(30):
        chemin_seq = os.path.join(chemin_geste, str(seq))
        if not os.path.exists(chemin_seq):
            continue
            
        for frame in range(30):
            img_path = os.path.join(chemin_seq, f"{frame}.jpg")
            
            if not os.path.exists(img_path):
                print(f"Fichier manquant : {img_path}")
                erreurs += 1
                continue
                
            img = cv2.imread(img_path)
            if img is None:
                print(f"Image corrompue : {img_path}")
                erreurs += 1
                continue
                
            if img.shape != (224, 224, 3):
                print(f"Mauvaise dimension {img.shape} : {img_path}")
                erreurs += 1
                continue
                
            total_images += 1

print("\n--- RÉSULTATS ---")
print(f"Images valides : {total_images} / 8100")
print(f"Erreurs        : {erreurs}")

if erreurs == 0 and total_images == 8100:
    print("Le dataset est parfait. Vous pouvez lancer l'entraînement !")
else:
    print("Le dataset est incomplet ou contient des erreurs.")