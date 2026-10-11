# 🚦 SafeCityAI — Console Autonome de Sécurité Routière & Détection d'Infractions

<div align="center">

  <img src="static/samples/sample_03.png" alt="SafeCityAI Preview Snapshot" width="720" style="border-radius: 12px; box-shadow: 0 8px 30px rgba(0,0,0,0.5);"/>

  <br/><br/>

  [![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
  [![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
  [![YOLOv5](https://img.shields.io/badge/YOLOv5-Ultralytics-00FFFF?style=for-the-badge&logo=opencv&logoColor=black)](https://github.com/ultralytics/yolov5)
  [![Flask](https://img.shields.io/badge/Flask-2.2%2B-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
  [![EasyOCR](https://img.shields.io/badge/EasyOCR-ALPR-4CAF50?style=for-the-badge&logo=googlelens&logoColor=white)](https://github.com/JaidedAI/EasyOCR)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
  [![Bilingual](https://img.shields.io/badge/Language-FR%20%F0%9F%87%AB%F0%9F%87%B7%20%7C%20EN%20%F0%9F%87%AC%F0%9F%87%A7-blue?style=for-the-badge)](#-système-bilingue--bilingual-ui)

  <p align="center">
    <strong>Plateforme Intelligente de Vidéosurveillance Routière, Inférence Double-Modèle YOLOv5 & Reconnaissance Optique de Plaques (ALPR).</strong>
    <br/>
    <em>Autonomous Road Safety & Traffic Enforcement System with Dual YOLOv5 Vision & ALPR Optical Recognition.</em>
  </p>

</div>

---

## 📑 Sommaire / Table of Contents

- [🇫🇷 Présentation du Projet (Français)](#-présentation-du-projet-français)
- [🇬🇧 Project Overview (English)](#-project-overview-english)
- [🧠 Architecture Double-Modèle & ALPR](#-architecture-double-modèle--alpr)
- [✨ Fonctionnalités Clés / Key Features](#-fonctionnalités-clés--key-features)
- [🌐 Système Bilingue / Bilingual UI](#-système-bilingue--bilingual-ui)
- [📁 Structure Complète du Projet](#-structure-complète-du-projet)
- [🚀 Démarrage Rapide / Quickstart](#-démarrage-rapide--quickstart)
- [🔌 Spécification de l'API REST](#-spécification-de-lapi-rest)
- [🧪 Tests Automatisés & Validation](#-tests-automatisés--validation)
- [📜 Licence & Copyright](#-licence--copyright)

---

## 🇫🇷 Présentation du Projet (Français)

**SafeCityAI** est une console de surveillance et de verbalisation automatisée conçue pour les services de sécurité routière et les municipalités intelligentes (*Smart Cities*).

Le système analyse les flux de circulation urbains pour intercepter en temps réel les infractions au Code de la Route (défaut de port du casque de protection obligatoire selon l'**Article R431-1**), isoler automatiquement les preuves photographiques par recadrage haute résolution, lire les plaques d'immatriculation (ALPR OCR) et générer instantanément un procès-verbal officiel prêt à l'emploi.

### Les 2 Modes Opérationnels Dédiés :

1. **📸 Mode Analyse Image (Contrôle Cliché)** :
   - Conçu pour le traitement de photos radar haute résolution ou l'audit a posteriori.
   - Découpage dynamique de **vignettes de preuves (Crops)** : tête/casque, corps du motard et plaque zoomée.
   - Reconnaissance optique des caractères (OCR) de la plaque et qualification légale du constat.
   - Procès-verbal officiel généré avec actions d'export et copie en 1 clic.

2. **🎥 Mode Vidéosurveillance Caméra (Direct 30 FPS)** :
   - Traitement continu d'un flux caméra de carrefour urbain (filtré à 100% sur les deux-roues).
   - Affichage HUD avec réticule numérique, horodatage synchronisé et bandeaux d'alerte en direct.
   - Journal chronologique des motards interceptés avec statut de conformité et matricule.
   - Simulateur 2D interactif et bouton de **capture de frame en 1 clic** basculant directement la scène en haute définition dans le Mode Analyse Image.

---

## 🇬🇧 Project Overview (English)

**SafeCityAI** is an autonomous traffic enforcement and safety compliance console engineered for road safety authorities and smart city infrastructure.

The platform continuously inspects urban traffic to detect non-compliant motorcyclists operating without safety helmets (mandated by traffic regulations), extracts high-precision visual proof crops, executes Automatic License Plate Recognition (ALPR OCR), and compiles official legal infraction dossiers in real-time.

---

## 🧠 Architecture Double-Modèle & ALPR

Pour garantir une précision maximale et éviter les compromis entre résolution photo et fluidité vidéo, SafeCityAI exploite une architecture neuronale découplée :

```
                        Flux Entrant (Photo ou Vidéo CCTV)
                                        │
                ┌───────────────────────┴───────────────────────┐
                ▼                                               ▼
       [ Mode Photo / Upload ]                       [ Flux Caméra CCTV Direct ]
                │                                               │
      Modèle: best_image.pt                          Modèle: best_cctv.pt
  (Précision mAP@0.5: 92.4% 🔒)                  (Spécialisé motos urbaines, 30 FPS)
  [best_image_backup.pt protégé]                                │
                │                                               │
                └───────────────────────┬───────────────────────┘
                                        ▼
                        Boîtes Englobantes (Bounding Boxes)
                          - 0: number plate
                          - 1: rider
                          - 2: with helmet
                          - 3: without helmet
                                        │
                                        ▼
                         Extraction de Zones d'Intérêt
                                (Visual Crops)
                                        │
                                        ▼
                          Moteur ALPR EasyOCR Local
                        (craft_mlt_25k + english_g2)
                                        │
                                        ▼
                      Dashboard Web & Procès-Verbal Officiel
```

* **`weights/best_image.pt`** : Entraîné pour l'analyse statique haute précision sur clichés fixes. Protégé par une sauvegarde miroir `weights/best_image_backup.pt`.
* **`weights/best_cctv.pt`** : Spécifiquement calibré pour les caméras de vidéosurveillance urbaine avec gestion du flou de mouvement et filtrage ciblé sur les motards.
* **`alpr_ocr.py`** : Moteur ALPR exploitant les réseaux de neurones profonds CRAFT (*Character Region Awareness for Text Detection*) et CRNN (*Convolutional Recurrent Neural Network*). Les modèles tournent 100% en local sans aucune dépendance cloud payante ni fuite de données privées.

---

## ✨ Fonctionnalités Clés / Key Features

- 🌐 **Interface Entièrement Bilingue (FR 🇫🇷 / EN 🇬🇧)** : Bascule instantanée sans rechargement de page avec mémorisation des préférences dans le navigateur (`localStorage`).
- 🔍 **Preuves Visuelles Découpées (Dynamic Crops)** : Découpage et zoom automatique sur la tête du conducteur, le motard et la plaque minéralogique pour vérification visuelle immédiate.
- 🏷️ **Cartouche Minéralogique ALPR Métallisé** : Affichage haute fidélité de la plaque extraite, prêt pour interconnexion au fichier national des véhicules.
- 📋 **Procès-Verbal Automatisé** : Rédaction en temps réel du constat juridique officiel (qualification légale, équipement constaté, immatriculation relevée, action requise).
- 🎥 **Contrôles Vidéo Complets** : Play / Pause du flux direct, bascule vers simulateur 2D cybernétique, capture instantanée de frame.
- 📊 **Tableau Télémétrique & Inspecteur JSON** : Liste exhaustive des coordonnées `[x, y, w, h]` et niveaux de confiance certifiés par l'IA.

---

## 🌐 Système Bilingue / Bilingual UI

Le sélecteur de langue situé dans l'en-tête supérieur permet d'adapter l'ensemble de la console à l'utilisateur :

| Élément d'interface | Version Française 🇫🇷 | Version Anglaise 🇬🇧 |
| :--- | :--- | :--- |
| **Sous-titre En-tête** | Console Autonome de Sécurité Routière & Verbalisation par IA | Autonomous Road Safety & AI Traffic Violation Enforcement Console |
| **Onglet 1** | Mode Analyse Image (Contrôle Cliché) | Image Analysis Mode (Snapshot Inspection) |
| **Onglet 2** | Mode Vidéosurveillance Caméra (Direct 30 FPS) | CCTV Video Surveillance Mode (Live 30 FPS) |
| **Verdict Infraction** | Défaut de Port du Casque de Protection Obligatoire | Failure to Wear Mandatory Safety Helmet |
| **Verdict Conforme** | Casque Réglementaire Porté & Homologué | Approved Helmet Worn & Verified |
| **Preuves Visuelles** | Preuves Visuelles Découpées (Zooms Haute Précision) | Visual Evidence Crops (High-Precision Zooms) |
| **Rapport Officiel** | Rapport Opérationnel pour Officier de Contrôle | Official Inspection Report for Enforcement Officer |

---

## 📁 Structure Complète du Projet

```text
safecity-ai/
├── alpr_ocr.py               # Moteur de reconnaissance optique de plaques (EasyOCR local)
├── model_engine.py           # Gestionnaire d'inférence double-modèle YOLOv5
├── server.py                 # Serveur Web Flask (API REST + Flux streaming MJPEG)
├── client_test.py            # Script client de test d'envoi d'image par requête HTTP POST
├── test_pipeline.py          # Script de validation complète sur l'ensemble des scénarios
├── train_cctv_isolated.py    # Entraînement isolé pour le modèle de caméra CCTV
├── extract_cctv_frames.py    # Découpage vidéo automatique en frames annotables
├── auto_annotate_cctv.py     # Pré-annotation assistée par modèle source
├── ccpd_tool.py              # Convertisseur de plaques CCPD pour enrichissement ALPR
│
├── datasets/                 # Bases de données YOLOv5 strictement partitionnées
│   ├── dataset_image/        # Clichés fixes (images + labels YOLO train/val)
│   ├── dataset_cctv/         # Flux vidéo CCTV (images + labels YOLO train/val)
│   └── README.md             # Documentation des jeux de données
│
├── weights/                  # Poids neuronaux pré-entraînés
│   ├── best_image.pt         # Modèle photo (92.4% mAP)
│   ├── best_image_backup.pt  # Sauvegarde sécurisée en lecture seule
│   ├── best_cctv.pt          # Modèle vidéosurveillance urbaine 30 FPS
│   └── README.md             # Guide d'architecture des modèles
│
├── static/                   # Ressources statiques frontend
│   ├── style.css             # Feuille de style CSS cyberpunk & glassmorphic
│   ├── app.js                # Contrôleur JS, gestion bilingue et rendu des preuves
│   └── samples/              # 6 Photos réelles de circulation pour test immédiat
│
├── templates/
│   └── index.html            # Interface web complète avec balisage sémantique
│
├── notebooks/                # Notebooks d'expérimentation et d'entraînement
│   ├── SafeCityAI_YOLOv5.ipynb
│   └── yolov5/               # Code source officiel Ultralytics YOLOv5
│
├── run_safecity.bat          # Lanceur automatique 1-clic pour Windows
├── start.sh                  # Lanceur multi-plateforme Linux / macOS
├── git_push.bat              # Script d'automatisation des commits & push GitHub
├── requirements.txt          # Dépendances Python du projet
├── LICENSE                   # Licence officielle MIT
├── .gitignore                # Exclusion des fichiers temporaires et médias lourds
└── README.md                 # Documentation principale du projet
```

---

## 🚀 Démarrage Rapide / Quickstart

### Prérequis
- **Python 3.10 ou supérieur** installé ([python.org](https://www.python.org/downloads/)).
- **Git** installé ([git-scm.com](https://git-scm.com/)).
- Navigateur web moderne (Chrome, Edge, Firefox, Brave).

### Option 1 : Lancement Automatique en 1 Clic (Recommandé)

* **Sous Windows :**
  Double-cliquez simplement sur le fichier :
  ```cmd
  run_safecity.bat
  ```

* **Sous Linux ou macOS :**
  Ouvrez un terminal et exécutez :
  ```bash
  chmod +x start.sh && ./start.sh
  ```

Le script configure automatiquement l'environnement virtuel, installe les dépendances nécessaires et démarre le serveur sur `http://localhost:5000`.

---

### Option 2 : Installation Manuelle via Terminal

1. **Cloner le dépôt :**
   ```bash
   git clone https://github.com/ratovoarisoaricardo/Safety-AI.git
   cd Safety-AI
   ```

2. **Créer et activer un environnement virtuel :**
   ```bash
   # Windows (PowerShell / CMD)
   python -m venv venv
   venv\Scripts\activate

   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Installer les dépendances :**
   ```bash
   pip install -r requirements.txt
   ```

4. **Lancer le serveur SafeCityAI :**
   ```bash
   python server.py
   ```

5. **Accéder à la console :**
   Ouvrez votre navigateur sur **[http://localhost:5000](http://localhost:5000)**.

---

## 🔌 Spécification de l'API REST

### 1. Inférence & Détection d'Infractions (`POST /api/detect`)
Permet de soumettre une image (multipart/form-data) et d'obtenir l'analyse complète au format JSON.

* **URL** : `http://localhost:5000/api/detect`
* **Méthode** : `POST`
* **Paramètres** :
  - `image` (Fichier binaire) : Image au format PNG, JPG ou JPEG.
  - `source_hint` *(Optionnel)* : `'photo'` (défaut) ou `'cctv_stream'`.

#### Exemple de Réponse JSON :
```json
{
  "status": "success",
  "detections": [
    {
      "box": [182, 94, 98, 106],
      "class": "without helmet",
      "confidence": 0.948
    },
    {
      "box": [120, 90, 240, 310],
      "class": "rider",
      "confidence": 0.912
    },
    {
      "box": [195, 340, 92, 44],
      "class": "number plate",
      "confidence": 0.895,
      "plate_text": "TN 09 BJ 4054"
    }
  ],
  "plate_detected": "TN 09 BJ 4054",
  "counts": {
    "total": 3,
    "rider": 1,
    "with helmet": 0,
    "without helmet": 1,
    "number plate": 1
  },
  "image_info": {
    "width": 640,
    "height": 480
  }
}
```

### 2. Flux Vidéo Direct CCTV (`GET /video/live_feed`)
* **URL** : `http://localhost:5000/video/live_feed`
* **Méthode** : `GET`
* **Type MIME** : `multipart/x-mixed-replace; boundary=frame`
* **Description** : Flux vidéo streaming MJPEG continu à 30 FPS avec boîtes englobantes dynamiques et matricules incrustés.

---

## 🧪 Tests Automatisés & Validation

Pour exécuter la suite de tests unitaires et vérifier la détection sur l'ensemble des scénarios de test :

```bash
python test_pipeline.py
```

Pour simuler une requête cliente complète vers l'API Flask :
```bash
python client_test.py
```

---

## 📜 Licence & Copyright

Ce projet est distribué sous les termes de la licence open-source **MIT**. Consultez le fichier [LICENSE](LICENSE) pour plus de détails.

```text
© 2026 SafeCityAI - Système Autonome de Détection et Sécurité Routière / Autonomous Road Safety & Traffic Enforcement System.
Tous droits réservés / All rights reserved.

Auteur & Développeur / Developer : ratovoarisoaricardo (Ratovoarisoa Mendrika Manjaka Ricardo)
Projet Vision par Ordinateur / Computer Vision Engineering Minor Project.
Architecture Double-Modèle YOLOv5 (best_image.pt 🔒 & best_cctv.pt) + ALPR EasyOCR.
```

<div align="center">
  <sub>Développé avec passion pour sauver des vies et moderniser la sécurité routière 🛵🚦🛡️</sub>
</div>