import os
import pickle
import argparse
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import json

def train_and_evaluate(train_path, test_path, output_path):
    """
    Charge les données d'entraînement et de test, entraîne un modèle RandomForest,
    évalue ses performances et sauvegarde les résultats ainsi que le modèle.
    """
    # Charger les caractéristiques et labels pour l'entraînement
    print("Chargement des données d'entraînement...")
    with open(train_path, 'rb') as f:
        X_train, y_train = pickle.load(f)
    
    # Charger les caractéristiques et labels pour le test
    print("Chargement des données de test...")
    with open(test_path, 'rb') as f:
        X_test, y_test = pickle.load(f)
    
    # Vérification des dimensions des données
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("Le nombre de caractéristiques dans les données d'entraînement et de test doit être identique.")
    
    # Entraînement du modèle
    print("Entraînement du modèle RandomForest...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Évaluation sur les données de test
    print("Évaluation du modèle sur les données de test...")
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error : {mse:.4f}")

    results = {
        'mean_squared_error': mse,
    }
    
    # Création du dossier de sortie si nécessaire
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Sauvegarder les résultats dans un fichier JSON
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f)
    print(f"Résultats sauvegardés dans {results_path}")

    # Sauvegarder le modèle entraîné
    model_path = os.path.join(output_dir, "model.pkl")
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)

    print(f"Modèle sauvegardé dans {model_path}")


if __name__ == '__main__':
    # Gestion des arguments
    parser = argparse.ArgumentParser(description="Entraînement et évaluation d'un modèle RandomForest.")
    parser.add_argument('--train', type=str, required=True, help="Chemin du fichier des données d'entraînement (Pickle)")
    parser.add_argument('--test', type=str, required=True, help="Chemin du fichier des données de test (Pickle)")
    parser.add_argument('--output', type=str, required=True, help="Dossier de sortie pour les résultats et le modèle")
    args = parser.parse_args()
    
    # Appel de la fonction principale
    train_and_evaluate(args.train, args.test, args.output)