# SafeCityAI - Gestion Sécurisée des Poids & Modèles

Cette structure empêche tout écrasement accidentel :

- **`best_image_backup.pt`** : Copie de sauvegarde originelle (protégée en Lecture Seule). NE JAMAIS SUPPRIMER.
- **`best_image.pt`** : Modèle actif pour le scan des photos et images du tableau de bord.
- **`best_cctv.pt`** : Modèle dédié aux vidéos CCTV. Dès qu'un fichier `best_cctv.pt` est placé ici, SafeCityAI l'associe automatiquement au flux de la caméra de circulation.
