import pickle

# Remplacez 'processed/labo_segments.pkl' par le chemin vers votre fichier .pkl
pkl_file_path = 'processed/lab_segments.pkl'

with open(pkl_file_path, 'rb') as file:
    data = pickle.load(file)

# Afficher le contenu
segments, labels = data

print("Nombre de segments:", len(segments))
print("Premier segment:", segments[0])
print("Nombre de labels:", len(labels))
print("Premier label:", labels[0])
