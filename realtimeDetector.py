import cv2
from keras.models import load_model
import numpy as np
import config

# Laden Sie Ihr trainiertes Modell
model = load_model(config.LOAD_MODEL_FROM_FILE)  # Ersetzen Sie 'Ihr_Modellpfad.h5' durch den tatsächlichen Pfad zu Ihrem Modell

# Emotionslabels aus dem FER2013-Dataset
EMOTIONS_LIST = ["Wütend", "Ekel", "Angst", "Glücklich", "Traurig", "Überrascht", "Neutral"]

# Initialisieren Sie die Kamera
cap = cv2.VideoCapture(0)

# Laden Sie die Haarcascade für die Gesichtserkennung
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    # Einzelbild der Kamera erfassen
    ret, frame = cap.read()
    if not ret:
        break

    # Konvertieren Sie das Bild in Graustufen
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Gesichter im Bild erkennen
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in faces:
        # Gesichtsausschnitt für die Emotionserkennung vorbereiten
        face_crop = gray[y:y+h, x:x+w]
        face_crop = cv2.resize(face_crop, (48, 48))
        face_crop = face_crop.astype('float32') / 255
        face_crop = np.expand_dims(face_crop, axis=0)
        face_crop = np.expand_dims(face_crop, axis=-1)

        # Emotionen vorhersagen
        predictions = model.predict(face_crop)
        max_index = np.argmax(predictions[0])
        most_likely_emotion = EMOTIONS_LIST[max_index]

        # Zeichnen Sie ein Rechteck um das Gesicht und beschriften Sie es mit der wahrscheinlichsten Emotion
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, most_likely_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Das Bild mit den Emotionen anzeigen
    cv2.imshow('Emotion Detector', frame)

    # Beenden, wenn 'q' gedrückt wird
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamera freigeben und Fenster schließen
cap.release()
cv2.destroyAllWindows()
