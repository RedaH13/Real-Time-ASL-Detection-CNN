import cv2

print("=== TEST WEBCAM SIMPLE ===")
print("Appuyez sur 'Q' pour quitter.")

# Ouverture du flux vidéo
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la caméra (Est-elle utilisée par une autre application ?)")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur : Impossible de lire l'image.")
        break

    # Affichage basique
    cv2.imshow('Test Camera Brut (OpenCV)', frame)

    # Quitter avec 'Q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✓ Test de la caméra terminé avec succès.")