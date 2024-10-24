import mne
import numpy as np
import pandas as pd

# Chemin du fichier EDF
edf_file = '/Users/mac/Documents/ITS/S5/Dispositif médical (Labiod)/Projet/VLA_VRW/lab/EEG/edf/1.edf'

# Charger le fichier EDF
raw_data = mne.io.read_raw_edf(edf_file, preload=True)

# Extraire les données dans un tableau NumPy
# get_data() renvoie les données sous forme de tableau NumPy
eeg_data = raw_data.get_data()

# Extraire les noms des canaux
channel_names = raw_data.ch_names

# Créer un DataFrame Pandas à partir des données NumPy
# Chaque canal devient une colonne, et les échantillons (valeurs temporelles) sont les lignes
df = pd.DataFrame(eeg_data.T, columns=channel_names)

# Afficher les premières lignes du DataFrame pour vérification
print(df.head())

