# SafeCityAI - Architecture des Bases de Données (Datasets)

Pour éviter tout conflit ou perte de modèle, les bases de données sont rigoureusement séparées :

## 1. `dataset_image/` (Base de Données Photos)
- **Rôle** : Utilisée pour la détection fine sur des photos fixes ou uploads individuels.
- **Modèle associé** : `weights/best_image.pt` (sauvegardé en `best_image_backup.pt`).

## 2. `dataset_cctv/` (Base de Données Vidéos & Caméras CCTV)
- **Rôle** : Utilisée pour les flux vidéo en continu, caméras IP de circulation, et gestion du flou de mouvement.
- **Fichier de configuration** : `datasets/dataset_cctv/data_cctv.yaml`
- **Modèle associé** : `weights/best_cctv.pt`
