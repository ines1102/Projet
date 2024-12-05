import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier

# Chemins vers vos fichiers
test_features_path = "features/reel_features.pkl"  # Test features
model_path = "results/model.pkl"  # Modèle entraîné (fichier model.pkl)
threshold = 0.5  # Seuil pour classifier les prédictions en binaire

# Charger les données de test (features et labels)
with open(test_features_path, 'rb') as file:
    X_test, y_test = pickle.load(file)

# Convertir y_test en tableau numpy pour éviter les erreurs
y_test = np.array(y_test)

# Charger le modèle entraîné
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Générer les prédictions sur les données de test
predictions = model.predict(X_test)

# Convertir les prédictions et labels en classes binaires
y_true_classes = (y_test >= threshold).astype(int)
y_pred_classes = (predictions >= threshold).astype(int)

# Vérification de la distribution des classes
unique, counts = np.unique(y_true_classes, return_counts=True)
print("Distribution des classes dans y_test :")
print(dict(zip(unique, counts)))

# Générer et afficher la matrice de confusion
cm = confusion_matrix(y_true_classes, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non Fatigue", "Fatigue"])
disp.plot(cmap="Blues")
plt.title("Matrice de confusion")
plt.show()
