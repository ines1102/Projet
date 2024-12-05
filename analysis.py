import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def analyze_results(model_path, results_path, labo_features_path, real_features_path):
    # Chargement des résultats
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Charger les caractéristiques utilisées pour les tests
    with open(labo_features_path, 'rb') as f:
        labo_features, _ = pickle.load(f)
    
    with open(real_features_path, 'rb') as f:
        real_features, _ = pickle.load(f)
    
    # Affichage de la matrice de confusion
    y_true = results["y_true"]  # Les vraies étiquettes
    y_pred = results["y_pred"]  # Les prédictions du modèle
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Classe 0", "Classe 1"], yticklabels=["Classe 0", "Classe 1"])
    plt.title("Matrice de confusion")
    plt.xlabel("Prédictions")
    plt.ylabel("Vraies étiquettes")
    plt.show()

    # Rapport de classification (précision, rappel, F1-score)
    report = classification_report(y_true, y_pred)
    print("Rapport de classification :")
    print(report)

    # Analyse des erreurs résiduelles
    errors = y_true - y_pred  # Erreurs résiduelles (si c'est un problème de régression)
    plt.figure(figsize=(8, 6))
    sns.histplot(errors, kde=True, color="red")
    plt.title("Erreur résiduelle")
    plt.xlabel("Erreur")
    plt.ylabel("Fréquence")
    plt.show()

    # Comparaison des performances sur les données de laboratoire vs réelles
    print("Comparaison des performances sur données laboratoire vs réelles :")
    # Par exemple, comparer les scores de précision sur labo et réel
    labo_accuracy = results["labo_accuracy"]
    real_accuracy = results["real_accuracy"]
    
    print(f"Précision sur les données labo : {labo_accuracy}")
    print(f"Précision sur les données réelles : {real_accuracy}")

    # Comparaison visuelle (si c'est une classification binaire, on peut afficher les courbes ROC par exemple)
    plt.figure(figsize=(8, 6))
    sns.lineplot(x=results["fpr_labo"], y=results["tpr_labo"], label="Labo")
    sns.lineplot(x=results["fpr_real"], y=results["tpr_real"], label="Réel")
    plt.title("Courbe ROC - Labo vs Réel")
    plt.xlabel("Taux de faux positifs")
    plt.ylabel("Taux de vrais positifs")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    model_path = "Github/Projet/results/model.pkl"  # Modèle entraîné
    results_path = "Github/Projet/results/results.json"  # Résultats du modèle
    labo_features_path = "Github/Projet/features/lab_features.pkl"  # Caractéristiques labo
    real_features_path = "Github/Projet/features/reel_features.pkl"  # Caractéristiques réel
    
    analyze_results(model_path, results_path, labo_features_path, real_features_path)