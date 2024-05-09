import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import config

DATA_LOCATION = config.NPZ_DATA_LOCATION
mergeddata_dict = np.load(DATA_LOCATION)

test_images = mergeddata_dict['2']
test_labels = mergeddata_dict['3']
test_images = test_images / 255.0
test_images = np.transpose(test_images, (2, 0, 1))
test_images = np.expand_dims(test_images, -1)

KERAS_DIRECTORY = config.KERAS_DIRECTORY

def evaluate_model(model_path, test_images, test_labels):
    try:
        model = keras.models.load_model(model_path)
        predictions = model.predict(test_images)
        predicted_labels = np.argmax(predictions, axis=1)
        accuracy = np.mean(predicted_labels == test_labels)
        return (model_path, accuracy, None, model)
    except OSError as e:
        return (model_path, None, f"Fehler beim Laden der Datei: {e}", None)

keras_files = os.listdir(KERAS_DIRECTORY)
results = []

for file in keras_files:
    model_path = os.path.join(KERAS_DIRECTORY, file)
    result = evaluate_model(model_path, test_images, test_labels)
    results.append(result)

# Filtere die Ergebnisse, um nur die Modelle mit verfügbarer Genauigkeit zu behalten
results = [result for result in results if result[1] is not None]

# Sortiere die verbleibenden Ergebnisse nach der Genauigkeit absteigend
results.sort(key=lambda x: x[1], reverse=True)

# Ausgabe der Ergebnisse
for model_path, accuracy, error, model in results:
    print(f"Modell {model_path}: Erfolgsquote = {accuracy}")

# Falls es mindestens ein Modell mit verfügbarer Genauigkeit gibt
if results:
    best_model_path, best_accuracy, best_error, best_model = results[0]
    print(f"\nDas beste Modell {best_model_path} hat die höchste Genauigkeit von {best_accuracy}.")
    if best_model:
        print("Details des besten Modells:")
        best_model.summary()  # Ausgabe der Modellzusammenfassung
else:
    print("Keine Modelle mit verfügbarer Genauigkeit gefunden.")