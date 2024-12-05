import os
import pickle
import argparse
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import json

def train_and_evaluate(train_path, test_path, output_path):
    # Charger les caractéristiques et labels pour l'entraînement
    with open(train_path, 'rb') as f:
        X_train, y_train = pickle.load(f)
    
    # Charger les caractéristiques et labels pour le test
    with open(test_path, 'rb') as f:
        X_test, y_test = pickle.load(f)
    
    # Entraînement du modèle
    print("Entraînement du modèle...")
    model = RandomForestRegressor(n_estimators=500, random_state=42)
    model.fit(X_train, y_train)
    
    # Évaluation sur les données de test
    print("Évaluation du modèle sur les données de test...")
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    results = {
        'mean_squared_error': mse,
    }
    
    # Création du dossier de sortie si nécessaire
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Sauvegarder les résultats dans un fichier JSON
    with open(output_path, 'w') as f:
        json.dump(results, f)
    print(f"Résultats sauvegardés dans {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, required=True, help="Données d'entraînement")
    parser.add_argument('--test', type=str, required=True, help="Données de test")
    parser.add_argument('--output', type=str, required=True, help="Fichier de sortie des résultats")
    args = parser.parse_args()
    
    train_and_evaluate(args.train, args.test, args.output)
