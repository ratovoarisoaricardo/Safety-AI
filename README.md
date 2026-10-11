# SafeCityAI 🚗🚦

<div align="center">
  <img src="static/samples/sample_car.jpg" alt="SafeCityAI Preview" width="600"/>
  <br />
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Flask-000000?style=flat-square&logo=flask&logoColor=white" />
  <img src="https://img.shields.io/badge/YOLOv5-00FFFF?style=flat-square&logo=opencv&logoColor=black" />
  <img src="https://img.shields.io/badge/License-MIT-green.svg?style=flat-square" />
</div>

---

## 🇬🇧 English

### 📖 About the Project
SafeCityAI is an autonomous, real-time traffic safety compliance monitoring and enforcement platform powered by a **Dual YOLOv5 Neural Architecture** and **ALPR OCR**. It provides distinct inspection modes for high-resolution photo scans and 30 FPS live CCTV video surveillance.

### ✨ Key Features
- 🌐 **Fully Bilingual UI (FR 🇫🇷 / EN 🇬🇧)**: Instantaneous dynamic language switcher with persistent user preference.
- 🧠 **Dual Neural Engine Architecture**:
  - `best_image.pt`: High-precision neural inference for static road camera captures.
  - `best_cctv.pt`: Real-time optimized detector filtered for urban motorcycle traffic.
- 🔍 **Visual Evidence Crops**: Automated high-precision zoom extractions (Helmet/Head, Rider, License Plate).
- 🏷️ **ALPR & OCR Reading**: Real-time license plate detection and optical character recognition with EasyOCR.
- 📋 **Enforcement Officer Reports**: Automated generation of legal violation dossiers (Art. R431-1).
- 🎥 **Dedicated CCTV Mode**: Live MJPEG video stream (30 FPS), interactive 2D simulation, and 1-click frame capture for forensic analysis.

### 💻 Getting Started
To get a local copy up and running, execute the appropriate command:

**Windows :**
```cmd
run_safecity.bat
```

**Linux / macOS :**
```bash
chmod +x start.sh && ./start.sh
```

**Manual / Standard CLI :**
```bash
python -m venv venv
# Windows : venv\Scripts\activate | Linux/macOS : source venv/bin/activate
pip install -r requirements.txt
python server.py
```

---

## 🇫🇷 Français

### 📖 À propos du projet
SafeCityAI est une plateforme autonome de surveillance et de verbalisation de la sécurité routière en temps réel, propulsée par une **architecture neuronale double YOLOv5** et un module **ALPR OCR**. Elle propose des modes distincts pour l'analyse photo haute résolution et la vidéosurveillance CCTV à 30 FPS.

### ✨ Fonctionnalités clés
- 🌐 **Interface Entièrement Bilingue (FR 🇫🇷 / EN 🇬🇧)** : Bascule dynamique instantanée avec mémorisation des préférences.
- 🧠 **Architecture Double-Moteur IA** :
  - `best_image.pt` : Détecteur haute précision pour clichés radar fixes (protégé en lecture seule).
  - `best_cctv.pt` : Modèle vidéo temps réel optimisé pour la circulation urbaine des deux-roues.
- 🔍 **Preuves Visuelles Découpées** : Découpage automatique des zones d'intérêt (Casque/Tête, Motard, Plaque).
- 🏷️ **Module ALPR & OCR** : Lecture optique de plaques minéralogiques en temps réel avec EasyOCR.
- 📋 **Rapport pour Agent de Contrôle** : Synthèse légale automatique pour verbalisation (Art. R431-1).
- 🎥 **Mode Vidéosurveillance CCTV Dédié** : Flux direct 30 FPS, simulateur 2D et capture de frame en 1 clic pour scan haute résolution.

### 🚀 Démarrage rapide

**Sous Windows :**
Double-cliquez simplement sur :
```cmd
run_safecity.bat
```

**Sous Linux / macOS :**
```bash
chmod +x start.sh && ./start.sh
```

---
<div align="center">
  <sub>Copyright &copy; 2026 Ratovoarisoa Mendrika Manjaka Ricardo (ratovoarisoaricardo) • All rights reserved.</sub>
</div>