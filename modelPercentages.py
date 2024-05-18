import utils
import os
import numpy as np
import keras
import config

def evaluate_model(model_path, test_images, test_labels):
    try:
        model = keras.models.load_model(model_path)
        # Überprüfe die Eingabeform des Modells
        input_shape = model.input_shape
        if input_shape != (None, 48, 48, 1):
            return (model_path, None, f"Falsche Eingabeform: erwartet {input_shape}, gefunden (None, 48, 48, 1)", None)
        
        predictions = model.predict(test_images)
        predicted_labels = np.argmax(predictions, axis=1)
        accuracy = np.mean(predicted_labels == test_labels)
        return (model_path, accuracy, None, model)
    except OSError as e:
        return (model_path, None, f"Fehler beim Laden der Datei: {e}", None)

def evaluate_models(dir: str = config.KERAS_DIRECTORY):
    
    # get test data
    _, _, test_images, test_labels = utils.get_data()

    # get files in model directory
    keras_files = os.listdir(dir)
    results = []

    # loop through model directory and evaluate each model and add the results to results
    for file in keras_files:
        try:
            model_path = os.path.join(dir, file)
            result = evaluate_model(model_path, test_images, test_labels)
            results.append(result)
        except ValueError:
            print(f'{file = }: has an undefined shape')
    
    results = filter_and_sort(results)
    print_results(results)

def filter_and_sort(results) -> list:
    # delete results with invalid/inexistent value
    results = [result for result in results if result[1] is not None]

    # sort results for accuracy in descending order
    results.sort(key=lambda x: x[1], reverse=True)
    return results

def print_results(results):
    # print accuracy for every model in results
    for model_path, accuracy, _, _ in results:
        print(f"Modell {model_path}: Erfolgsquote = {accuracy}")

    # print results and summary for the best model in results
    if results:
        best_model_path, best_accuracy, _, best_model = results[0]
        print(f"\nDas beste Modell {best_model_path} hat die höchste Genauigkeit von {best_accuracy}.")
        if best_model:
            print("Details des besten Modells:")
            best_model.summary()  # Ausgabe der Modellzusammenfassung
    else:
        print("Keine Modelle mit verfügbarer Genauigkeit gefunden.")

if __name__ == '__main__':
    evaluate_models()
