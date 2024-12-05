import pickle
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Charger les caractéristiques
with open("features/lab_features.pkl", "rb") as f:
    features, labels = pickle.load(f)

# Normaliser les caractéristiques
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Séparer les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, labels, test_size=0.2, random_state=42)

# Entraîner le modèle
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Prédictions
y_pred = model.predict(X_test)

# Calculer la MSE
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")

# Calculer le coefficient de détermination R^2
r2 = r2_score(y_test, y_pred)
print(f"R^2: {r2}")

# Sauvegarder le modèle
with open("results/model.pkl", "wb") as f:
    pickle.dump(model, f)

# Sauvegarder les résultats
results = {
    "mean_squared_error": mse,
    "r2_score": r2,
    "y_true": y_test.tolist(),  # Convertir en liste pour JSON
    "y_pred": y_pred.tolist()   # Convertir en liste pour JSON
}

with open("results/results.json", "w") as f:
    json.dump(results, f, indent=4)

print("Résultats sauvegardés dans 'results.json'")

# Visualisation des résultats
plt.scatter(y_test, y_pred)
plt.xlabel('Valeurs réelles')
plt.ylabel('Prédictions')
plt.title('Comparaison entre valeurs réelles et prédictions')
plt.show()