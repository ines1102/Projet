import json
import matplotlib.pyplot as plt

# Chemin vers le fichier JSON
results_path = "Github/Projet/results/results.json"

# Charger les résultats
with open(results_path, "r") as f:
    results = json.load(f)

# Extraire la MSE
mse = results.get("mean_squared_error", None)
print(f"Erreur quadratique moyenne (MSE) : {mse:.6f}" if mse else "MSE non trouvée.")

# Vérifier si y_true et y_pred sont disponibles
y_true = results.get("y_true", None)
y_pred = results.get("y_pred", None)

if y_true and y_pred:
    # Générer un graphique comparatif
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label="Valeurs réelles", marker='o')
    plt.plot(y_pred, label="Prédictions", marker='x')
    plt.xlabel("Index")
    plt.ylabel("Valeurs")
    plt.title("Comparaison des valeurs réelles et des prédictions")
    plt.legend()
    plt.grid()
    plt.show()
else:
    print("Les données 'y_true' et 'y_pred' sont absentes du fichier JSON.")