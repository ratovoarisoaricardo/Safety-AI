# ==============================================================================
# Safety AI - Script d'Entraînement Isolé & Sécurisé pour Caméra CCTV
# Ce script garantit que le modèle 'best_image.pt' ne sera JAMAIS touché ni écrasé.
# ==============================================================================

import os
import sys
import shutil
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKUP_PATH = os.path.join(BASE_DIR, 'weights', 'best_image_backup.pt')
IMAGE_MODEL_PATH = os.path.join(BASE_DIR, 'weights', 'best_image.pt')
DATA_YAML = os.path.join(BASE_DIR, 'datasets', 'dataset_cctv', 'data_cctv.yaml')
RUNS_DIR = os.path.join(BASE_DIR, 'runs_cctv')
CCTV_DEST_PATH = os.path.join(BASE_DIR, 'weights', 'best_cctv.pt')

def verify_safety():
    print("=" * 70)
    print(" [SÉCURITÉ] Vérification préalable de l'intégrité des modèles...")
    print("=" * 70)
    
    if os.path.exists(BACKUP_PATH):
        print(f" [OK] Sauvegarde blindée présente : {BACKUP_PATH} ({os.path.getsize(BACKUP_PATH)} octets)")
    else:
        print(" [ATTENTION] Création d'urgence de la sauvegarde sécurisée...")
        os.makedirs(os.path.dirname(BACKUP_PATH), exist_ok=True)
        if os.path.exists(IMAGE_MODEL_PATH):
            shutil.copy2(IMAGE_MODEL_PATH, BACKUP_PATH)

    if not os.path.exists(DATA_YAML):
        print(f" [ERREUR] Le fichier YAML BDD CCTV est introuvable : {DATA_YAML}")
        return False

    print(" [OK] La base de données CCTV est bien isolée dans 'datasets/dataset_cctv'.")
    print(" [OK] Les résultats d'entraînement seront stockés dans 'runs_cctv/' sans toucher à la racine.")
    return True

def run_isolated_training(epochs=20, batch=8, img_size=640):
    if not verify_safety():
        return

    print("\n" + "=" * 70)
    print(f" [ENTRAÎNEMENT] Lancement sur BDD CCTV (Epochs: {epochs}, Batch: {batch})...")
    print("=" * 70)

    # Commande YOLOv5 entièrement isolée
    yolo_train = os.path.join(BASE_DIR, 'notebooks', 'yolov5', 'train.py')
    if not os.path.exists(yolo_train):
        yolo_train = 'train.py'

    cmd = [
        sys.executable, yolo_train,
        '--img', str(img_size),
        '--batch', str(batch),
        '--epochs', str(epochs),
        '--data', DATA_YAML,
        '--weights', 'yolov5s.pt',
        '--project', RUNS_DIR,
        '--name', 'cctv_experiment',
        '--exist-ok'
    ]

    print(f" [COMMANDE] {' '.join(cmd)}")
    print(" [INFO] Démarrage du processus isolé...")
    
    try:
        subprocess.run(cmd, check=True)
        
        # Récupération automatique et sécurisée du nouveau modèle
        trained_best = os.path.join(RUNS_DIR, 'cctv_experiment', 'weights', 'best.pt')
        if os.path.exists(trained_best):
            shutil.copy2(trained_best, CCTV_DEST_PATH)
            print("\n" + "=" * 70)
            print(" [FÉLICITATIONS] Entraînement CCTV terminé avec succès !")
            print(f" -> Nouveau modèle vidéo enregistré dans : {CCTV_DEST_PATH}")
            print(f" -> Votre ancien modèle 'best_image.pt' est 100% INTACT et inchangé.")
            print("=" * 70)
        else:
            print(" [INFO] Entraînement terminé, aucun nouveau poids 'best.pt' généré.")
            
    except Exception as e:
        print(f" [AVERTISSEMENT] Processus d'entraînement interrompu ou non terminé : {e}")

if __name__ == '__main__':
    run_isolated_training()
