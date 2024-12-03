import pickle

# Remplacez 'processed/labo_segments.pkl' par le chemin vers votre fichier .pkl
pkl_file_path = 'processed/reel_segments.pkl'

with open(pkl_file_path, 'rb') as file:
    data = pickle.load(file)

# Vérification des données chargées
if not data:
    print("Erreur : Les données chargées sont vides.")
else:
    segments, labels = data
    print("Nombre de segments:", len(segments))
    if len(segments) > 0:
        print("Premier segment:", segments[0])
    print("Nombre de labels:", len(labels))
    if len(labels) > 0:
        print("Premier label:", labels[0])
