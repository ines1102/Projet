import pickle
import argparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score

def train_and_evaluate(train_path, test_path, output_path):
    with open(train_path, 'rb') as f:
        X_train, y_train = pickle.load(f)
    
    with open(test_path, 'rb') as f:
        X_test, y_test = pickle.load(f)
    
    # Entraînement du modèle
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Évaluation
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    results = {
        'mean_squared_error': mse,
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, required=True, help="Données d'entraînement")
    parser.add_argument('--test', type=str, required=True, help="Données de test")
    parser.add_argument('--output', type=str, required=True, help="Fichier de sortie des résultats")
    args = parser.parse_args()
    
    train_and_evaluate(args.train, args.test, args.output)
