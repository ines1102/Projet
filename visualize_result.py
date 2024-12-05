import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# Charger le modèle entraîné et les données de test
model_path = "results/model.pkl"
test_data_path = "features/reel_features.pkl"

# Charger le modèle sauvegardé
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Charger les données de test (features et labels)
with open(test_data_path, 'rb') as f:
    X_test, y_test = pickle.load(f)

# Prédire les valeurs avec le modèle
predictions = model.predict(X_test)

# Calculer l'erreur quadratique moyenne (MSE) et le R²
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# Afficher les métriques de performance
print(f"Erreur quadratique moyenne (MSE) : {mse}")
print(f"R² : {r2}")

# --- 1. Graphique des résidus ---
residuals = y_test - predictions

plt.figure(figsize=(10, 6))
plt.scatter(predictions, residuals, color='blue', alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.title("Graphique des erreurs résiduelles")
plt.xlabel("Prédictions")
plt.ylabel("Résidus")
plt.show()

# --- 2. Prédictions vs Valeurs réelles ---
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, color='blue', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title("Prédictions vs. Valeurs réelles")
plt.xlabel("Valeurs réelles")
plt.ylabel("Prédictions")
plt.show()

# --- 3. Importance des features ---
importances = model.feature_importances_

# Trier les features par importance
indices = np.argsort(importances)[::-1]

# Afficher l'importance des features
plt.figure(figsize=(10, 6))
plt.title("Importance des features")
plt.barh(range(X_test.shape[1]), importances[indices], align="center")
plt.yticks(range(X_test.shape[1]), [f"Feature {i}" for i in indices])
plt.xlabel("Importance")
plt.show()