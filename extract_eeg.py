import mne
import numpy as np
import matplotlib.pyplot as plt

# Charger le fichier EDF
file_path = "/Users/mac/Documents/ITS/S5/Dispositif médical (Labiod)/Projet/VLA_VRW/lab/EEG/edf/1.edf"
raw_eeg = mne.io.read_raw_edf(file_path, preload=True)

# Filtrer le signal à 40 Hz
raw_eeg.filter(1, 40)

# Extraire les données EEG
eeg_data = raw_eeg.get_data()
sf = raw_eeg.info['sfreq']  # Fréquence d'échantillonnage

# Visualiser le signal pour vérifier les problèmes éventuels
plt.plot(eeg_data[0, :1000])  # Visualiser les premiers échantillons du premier canal
plt.title("Visualisation du signal brut")
plt.show()

# Vérifier s'il y a des NaN ou des valeurs aberrantes dans les données
if np.isnan(eeg_data).any():
    print("Les données contiennent des valeurs NaN. Interpolation en cours...")
    eeg_data = np.nan_to_num(eeg_data, nan=np.nanmean(eeg_data))

# Calculer la PSD avec des paramètres ajustés
psd, freqs = mne.time_frequency.psd_array_multitaper(
    eeg_data, sf, adaptive=False, normalization='length', verbose=True
)

# Afficher la puissance dans la bande Alpha
alpha_band = (8, 12)  # Définir la bande Alpha
alpha_power = np.sum(psd[:, (freqs >= alpha_band[0]) & (freqs <= alpha_band[1])], axis=1)
print(f"Puissance dans la bande Alpha: {alpha_power}")