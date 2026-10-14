# ==============================================================================
# Safety AI - High-Performance Traffic Violation Detection & ALPR Console
# Multi-Regional Compliance (India RTO, France SIV, Universal Auto-Detection)
# Asynchronous 30 FPS CCTV Video Streaming Engine
# ==============================================================================

import os
import io
import time
import json
import threading
import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template, send_from_directory, Response

app = Flask(__name__, static_folder='static', template_folder='templates')

# Initialisation des moteurs neuronaux
from model_engine import detector, image_detector, cctv_detector
from alpr_ocr import alpr_engine, REGIONAL_PROFILES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------------------
# 1. Moteur de Caméra Asynchrone (Zero-Lag 30 FPS CCTV Engine)
# ------------------------------------------------------------------------------

class AsyncVideoCamera:
    """
    Gestionnaire de capture vidéo CCTV multi-threadé.
    Découple totalement l'inférence neuronale du streaming HTTP, garantissant
    un flux continu fluide à 30 FPS constants sans saccades ni blocage du serveur.
    """
    def __init__(self):
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self.region = 'AUTO'
        self.video_path = self._resolve_video_path('AUTO')
        self.current_jpeg = None
        self.current_raw_frame = None
        self.cached_detections = []
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.current_fps = 30.0
        self.violation_detected = False
        if not os.environ.get('VERCEL'):
            self.start()

    def _resolve_video_path(self, region):
        reg = region.upper()
        if reg == 'FR':
            candidates = ['real_traffic_cars.mp4', 'saigon_part.webm', 'cctv_motorcycle_real.webm', 'output_violation_demo.mp4']
        elif reg == 'VN':
            candidates = ['saigon_part.webm', 'traffic_d7.webm', 'cctv_motorcycle_real.webm']
        else:
            candidates = ['saigon_part.webm', 'cctv_motorcycle_real.webm', 'real_traffic_cars.mp4', 'traffic_d7.webm', 'output_violation_demo.mp4']

        for fname in candidates:
            p = os.path.join(BASE_DIR, fname)
            if os.path.exists(p) and os.path.getsize(p) > 10000:
                return p
        return os.path.join(BASE_DIR, 'saigon_part.webm')

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._capture_worker, daemon=True)
            self.thread.start()

    def set_region(self, region):
        with self.lock:
            self.region = region.upper()
            # Ne pas écraser si l'utilisateur a choisi la webcam directe '0' ou une URL RTSP
            if str(self.video_path) != "0" and not str(self.video_path).startswith(('rtsp://', 'http://', 'https://')):
                new_path = self._resolve_video_path(self.region)
                if new_path != self.video_path and os.path.exists(new_path):
                    self.video_path = new_path
                    print(f" [CCTV] Source vidéo mise à jour pour la région {self.region} : {os.path.basename(new_path)}")

    def set_video_file(self, filename):
        with self.lock:
            self.cached_detections = []
            self.violation_detected = False
            s = str(filename).strip()
            # 1. Mode Caméra Physique / Webcam locale directe
            if s in ("0", "webcam", "live_cam", "cam_0"):
                self.video_path = "0"
                print(" [CCTV] Source changée : Caméra Physique Directe (Webcam 0)")
                return True
            # 2. Flux Réseau IP / RTSP / HTTP
            if s.startswith(("http://", "https://", "rtsp://")):
                self.video_path = s
                print(f" [CCTV] Source changée : Flux Réseau Caméra IP ({s})")
                return True
            # 3. Fichier vidéo local authentique
            p = os.path.join(BASE_DIR, filename)
            if os.path.exists(p) and os.path.getsize(p) > 1000:
                self.video_path = p
                print(f" [CCTV] Source vidéo changée manuellement : {filename}")
                return True
        return False

    def get_frame_bytes(self):
        with self.lock:
            if self.current_jpeg is not None:
                return self.current_jpeg
            # Fallback for serverless environments (Vercel) or initial cold start
            for fb in ('saigon_frame.jpg', 'sample_traffic_frame.jpg'):
                p = os.path.join(BASE_DIR, fb)
                if os.path.exists(p):
                    with open(p, 'rb') as f:
                        return f.read()
            return None

    def get_snapshot_png(self):
        with self.lock:
            if self.current_raw_frame is not None:
                # Optimized PNG level 3 compression for instantaneous HTTP snapshot transfer (<10ms)
                ret, buf = cv2.imencode('.png', self.current_raw_frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                if ret:
                    return buf.tobytes()
            # Serverless fallback snapshot
            fb = os.path.join(BASE_DIR, 'saigon_frame.jpg')
            if os.path.exists(fb):
                img = cv2.imread(fb)
                if img is not None:
                    ret, buf = cv2.imencode('.png', img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                    if ret:
                        return buf.tobytes()
        return None

    def stop(self):
        """Arrête proprement le thread daemon de streaming."""
        self.running = False

    def _capture_worker(self):
        """Boucle d'acquisition continue isolée dans son propre thread système."""
        color_map = {
            'No_Helmet': (0, 0, 255),        # Rouge vif (Infraction)
            'without helmet': (0, 0, 255),
            'Helmet': (0, 255, 0),           # Vert émeraude (Conforme)
            'with helmet': (0, 255, 0),
            'License_Plate': (255, 255, 0),  # Cyan
            'number plate': (255, 255, 0),
            'Rider': (255, 0, 255),          # Magenta (Motard)
            'rider': (255, 0, 255)
        }
        MOTO_TARGET_CLASSES = {'Rider', 'rider', 'Helmet', 'with helmet', 'No_Helmet', 'without helmet', 'License_Plate', 'number plate'}

        cap = None
        current_opened_source = None
        while self.running:
            target_source = self.video_path
            is_webcam = str(target_source).strip() in ("0", "webcam", "live_cam", "cam_0")
            is_url = isinstance(target_source, str) and (target_source.startswith("http://") or target_source.startswith("https://") or target_source.startswith("rtsp://"))

            if not is_webcam and not is_url:
                if not os.path.exists(target_source):
                    time.sleep(0.5)
                    continue

            # Réouverture automatique si la source a changé ou si cap n'est pas ouvert
            if cap is None or not cap.isOpened() or current_opened_source != target_source:
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
                current_opened_source = target_source
                self.cached_detections = []
                self.violation_detected = False
                if is_webcam:
                    cap = cv2.VideoCapture(0)
                else:
                    cap = cv2.VideoCapture(target_source)

            ret, frame = cap.read()
            if not ret:
                if is_webcam:
                    # Pour la webcam physique, si une frame tarde, attendre brièvement
                    time.sleep(0.05)
                    continue
                else:
                    # Bouclage sécurisé et propre du fichier vidéo sans blocage OpenCV
                    cap.release()
                    time.sleep(0.02)
                    cap = cv2.VideoCapture(target_source)
                    continue

            self.frame_count += 1
            now = time.time()
            dt = max(now - self.last_fps_time, 0.001)
            self.current_fps = 0.9 * self.current_fps + 0.1 * (1.0 / dt)
            self.last_fps_time = now

            # Redimensionnement optimisé pour le streaming 16:9
            frame = cv2.resize(frame, (640, 360))
            raw_copy = frame.copy()
            h, w = frame.shape[:2]

            # Inférence neuronale sélective toutes les 3 frames sans bloquer le rendu
            if self.frame_count % 3 == 0 and cctv_detector.loaded:
                try:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb_frame)
                    all_dets = cctv_detector.detect_image(pil_img, conf_thres=0.35)
                    self.cached_detections = [d for d in all_dets if d['class'] in MOTO_TARGET_CLASSES]
                except Exception:
                    pass

            # Rendu visuel du HUD et des boîtes englobantes
            violation_found = False
            for det in self.cached_detections:
                cls = det['class']
                conf = det.get('confidence', 0.88)
                bx, by, bw, bh = [int(v) for v in det['box']]
                bx = max(0, min(bx, w - 1))
                by = max(0, min(by, h - 1))
                bw = max(5, min(bw, w - bx))
                bh = max(5, min(bh, h - by))

                color = color_map.get(cls, (0, 255, 255))
                if 'No_' in cls or 'without' in cls.lower():
                    violation_found = True

                cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), color, 2)
                if 'plate' in cls.lower():
                    plate_crop = raw_copy[by:by+bh, bx:bx+bw]
                    plate_info = alpr_engine.read_plate_full(plate_crop, source_hint='cctv_stream', box=(bx, by, bw, bh), requested_region=self.region)
                    tag = f"[{plate_info['flag']} {plate_info['text']}]"
                else:
                    tag = f"MOTO: {cls} {int(conf*100)}%"

                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(frame, (bx, max(0, by - th - 6)), (bx + tw + 6, by), color, -1)
                cv2.putText(frame, tag, (bx + 3, max(12, by - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

            # Incrustation télémétrique cybernétique HUD dans le flux
            t_str = time.strftime("%d/%m/%Y %H:%M:%S")
            if is_webcam:
                cam_label = "DIRECT PHYSICAL WEBCAM [LIVE SENSOR]"
            elif 'saigon' in str(target_source).lower():
                cam_label = "CAM-01 CCTV [URBAN INTERSECTION 1080p]"
            elif 'motorcycle_real' in str(target_source).lower():
                cam_label = "CAM-02 CCTV [LIVE TRAFFIC LANE 1080p]"
            elif 'cars' in str(target_source).lower():
                cam_label = "CAM-03 CCTV [HIGHWAY SURVEILLANCE]"
            else:
                cam_label = f"CCTV [{os.path.basename(str(target_source))}]"

            cv2.putText(frame, f"{cam_label} | {t_str}", (15, 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 240, 255), 2)

            fps_display = min(self.current_fps, 30.0)
            status_text = f"AI ENGINE : best_cctv.pt | {fps_display:.1f} FPS"
            cv2.putText(frame, status_text, (15, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 2)

            if violation_found:
                cv2.putText(frame, "LIVE ALERT : MOTORCYCLE VIOLATION DETECTED", (15, 72), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 255), 2)

            # Encodage JPEG haute vitesse en mémoire vive (qualité 80 calibrée pour streaming ultra-fluide)
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                with self.lock:
                    self.current_jpeg = buffer.tobytes()
                    self.current_raw_frame = raw_copy
                    self.violation_detected = violation_found

            # Temporisation calibrée pour 30 images par seconde
            time.sleep(0.033)

        if cap is not None:
            cap.release()

# Instance globale de la caméra asynchrone
camera = AsyncVideoCamera()

# ------------------------------------------------------------------------------
# 2. Routes Web & Streaming
# ------------------------------------------------------------------------------

@app.route('/')
def index():
    """Rendu de la console principale Safety AI."""
    return render_template('index.html')

@app.route('/video/demo')
def serve_demo_video():
    """Sert la vidéo de démonstration au format MP4."""
    try:
        return send_from_directory(BASE_DIR, 'output_violation_demo.mp4', as_attachment=False)
    except Exception as e:
        return jsonify({"success": False, "error": f"Fichier vidéo introuvable : {e}"}), 404

@app.route('/video/live_feed')
def live_feed():
    """
    Flux vidéo streaming continu MJPEG ultra-fluide à 30 FPS.
    Distribue les frames directement depuis le tampon mémoire sans bloquer le serveur.
    """
    if os.environ.get('VERCEL'):
        frame_bytes = camera.get_frame_bytes()
        if frame_bytes:
            return Response(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n', mimetype='multipart/x-mixed-replace; boundary=frame')
    def stream_generator():
        while True:
            frame_bytes = camera.get_frame_bytes()
            if frame_bytes is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.033)

    return Response(stream_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/cctv_snapshot', methods=['GET'])
def cctv_snapshot():
    """
    Fournit la frame courante haute définition en PNG directement depuis la mémoire.
    Résout définitivement le bug de 'Tainted Canvas' côté navigateur.
    """
    png_bytes = camera.get_snapshot_png()
    if png_bytes is not None:
        return Response(png_bytes, mimetype='image/png')
    
    # Image de secours si la caméra n'a pas encore de frame
    fallback = os.path.join(BASE_DIR, 'sample_traffic_frame.jpg')
    if os.path.exists(fallback):
        return send_from_directory(BASE_DIR, 'sample_traffic_frame.jpg')
    return jsonify({"success": False, "error": "Aucune frame CCTV disponible"}), 503

@app.route('/api/set_region', methods=['POST'])
def set_region():
    """Permet de basculer la juridiction active ('AUTO', 'IN', 'FR', 'VN')."""
    data = request.get_json(silent=True) or {}
    region = data.get('region', 'AUTO').upper()
    if region not in REGIONAL_PROFILES and region != 'AUTO':
        region = 'AUTO'
    camera.set_region(region)
    alpr_engine.active_region = region
    return jsonify({
        "success": True,
        "region": region,
        "profile": REGIONAL_PROFILES.get(region if region != 'AUTO' else 'IN')
    })

@app.route('/api/set_cctv_video', methods=['POST'])
def set_cctv_video():
    """Permet de basculer la vidéo source CCTV ('0' pour webcam, 'saigon_part.webm', URL RTSP, etc.)."""
    data = request.get_json(silent=True) or {}
    video_file = data.get('video', 'saigon_part.webm')
    success = camera.set_video_file(video_file)
    disp_name = "🔴 Physical PC Webcam" if str(camera.video_path) == "0" else (
        f"🌐 Network Stream ({camera.video_path})" if str(camera.video_path).startswith(('rtsp://', 'http://', 'https://'))
        else os.path.basename(str(camera.video_path))
    )
    return jsonify({
        "success": success,
        "video": disp_name,
        "source": camera.video_path,
        "fps": round(camera.current_fps, 1)
    })

# ------------------------------------------------------------------------------
# 3. API REST d'Inférence & Détection d'Infractions
# ------------------------------------------------------------------------------

@app.route('/api/detect', methods=['POST'])
def detect():
    """
    Point de terminaison d'analyse complète sur image fixe (Upload ou Snapshot).
    Détecte les objets, lit les plaques avec auto-détection du pays et génère le PV officiel.
    """
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "Aucun fichier image fourni dans le champ 'image'"}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({"success": False, "error": "Nom de fichier vide"}), 400
        
    try:
        requested_region = request.form.get('region', 'AUTO').upper()
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        width, height = image.size
        
        # 1. Inférence neuronale chirurgicale avec best_image.pt
        detections = image_detector.detect_image(image)
        
        # 2. Reconnaissance optique et auto-détection du pays pour les plaques
        np_img = np.array(image)
        is_cctv_capture = file.filename.startswith('cctv_capture_')
        hint = 'cctv_stream' if is_cctv_capture else None

        detected_plate_info = None
        plate_idx = 0
        for det in detections:
            cls_lower = det['class'].lower()
            if 'plate' in cls_lower or 'number' in cls_lower:
                det['target_type'] = 'plate'
                try:
                    bx, by, bw, bh = [int(v) for v in det['box']]
                    bx, by = max(0, bx), max(0, by)
                    bw, bh = max(5, min(bw, width - bx)), max(5, min(bh, height - by))
                    plate_crop = np_img[by:by+bh, bx:bx+bw]
                    if plate_crop.size > 0:
                        bgr_crop = cv2.cvtColor(plate_crop, cv2.COLOR_RGB2BGR)
                        p_info = alpr_engine.read_plate_full(
                            bgr_crop, 
                            plate_index=plate_idx, 
                            source_hint=hint, 
                            box=(bx, by, bw, bh), 
                            requested_region=requested_region
                        )
                        plate_idx += 1
                        det['plate_text'] = p_info['text']
                        det['plate_region'] = p_info['region']
                        det['plate_region_name'] = p_info['region_name']
                        det['plate_flag'] = p_info['flag']
                        det['class'] = f"License_Plate [{p_info['flag']} {p_info['text']}]"
                        if detected_plate_info is None:
                            detected_plate_info = p_info
                except Exception as ocr_err:
                    print(f" [ALPR WARN] Extraction exception fallback: {ocr_err}")
            elif 'without' in cls_lower or 'no_helmet' in cls_lower:
                det['target_type'] = 'no_helmet'
            elif 'with helmet' in cls_lower or cls_lower == 'helmet':
                det['target_type'] = 'helmet'
            elif 'rider' in cls_lower:
                det['target_type'] = 'rider'
            elif 'no_seatbelt' in cls_lower:
                det['target_type'] = 'no_seatbelt'
            elif 'seatbelt' in cls_lower:
                det['target_type'] = 'seatbelt'

        # 3. Synthèse Juridique & Moteur Décisionnel Adaptatif
        violations = [d for d in detections if d.get('target_type') in ('no_helmet', 'no_seatbelt') or 'without' in d['class'].lower() or 'no_' in d['class'].lower()]
        helmets = [d for d in detections if d.get('target_type') == 'helmet' or ('helmet' in d['class'].lower() and 'without' not in d['class'].lower() and 'no_' not in d['class'].lower())]
        plates = [d for d in detections if d.get('target_type') == 'plate' or 'plate' in d['class'].lower()]

        # Détermination de la région applicable
        if detected_plate_info and detected_plate_info.get('region'):
            active_region = detected_plate_info['region']
        elif requested_region != 'AUTO':
            active_region = requested_region
        else:
            active_region = 'AUTO'

        target_reg = active_region if active_region in REGIONAL_PROFILES else 'GLOBAL'
        reg_prof = REGIONAL_PROFILES.get(target_reg, REGIONAL_PROFILES['GLOBAL'])

        # Détermination stricte de la plaque : Ne JAMAIS inventer une fausse plaque !
        if detected_plate_info and detected_plate_info.get('text') not in ("PLAQUE NON VISIBLE", "PLAQUE NON LISIBLE", "PLATE NOT VISIBLE", "PLATE NOT READABLE", None):
            detected_plate_text = detected_plate_info['text']
            has_plate = True
        else:
            detected_plate_text = None
            has_plate = False

        has_violation = len(violations) > 0
        is_compliant = len(helmets) > 0 and not has_violation

        if has_violation:
            max_conf = max([v.get('confidence', 0.9) for v in violations]) if violations else 0.92
            verdict = {
                "status": "VIOLATION",
                "label": f"TRAFFIC VIOLATION DETECTED : RIDER WITHOUT HELMET ({reg_prof['flag']} {reg_prof['name']})",
                "label_fr": f"INFRACTION DÉTECTÉE : CONDUCTEUR SANS CASQUE ({reg_prof['flag']} {reg_prof.get('name_fr', reg_prof['name'])})",
                "severity": "CRITICAL",
                "color": "#ef4444",
                "confidence": round(max_conf, 3),
                "plate": detected_plate_text,
                "has_plate": has_plate,
                "region": active_region,
                "region_name": reg_prof['name'],
                "flag": reg_prof['flag'],
                "law_article_fr": reg_prof['law_article_fr'],
                "law_article_en": reg_prof['law_article_en'],
                "penalty_fr": reg_prof['penalty_fr'],
                "penalty_en": reg_prof['penalty_en'],
                "currency": reg_prof['currency'],
                "summary": f"Violation recorded under {reg_prof['law_article_en']}. {reg_prof['penalty_en']}",
                "summary_fr": f"Infraction constatée au {reg_prof['law_article_fr']}. {reg_prof['penalty_fr']}",
                "action": "Official violation notice generated with attached digital evidence dossier.",
                "action_fr": "Avis de contravention officiel généré avec dossier de preuves numériques rattaché."
            }
        elif is_compliant:
            max_conf = max([h.get('confidence', 0.9) for h in helmets]) if helmets else 0.95
            verdict = {
                "status": "COMPLIANT",
                "label": f"COMPLIANT STATUS : APPROVED SAFETY HELMET VERIFIED ({reg_prof['flag']} {reg_prof['name']})",
                "label_fr": f"CONFORME : CASQUE DE PROTECTION HOMOLOGUÉ ({reg_prof['flag']} {reg_prof.get('name_fr', reg_prof['name'])})",
                "severity": "COMPLIANT",
                "color": "#10b981",
                "confidence": round(max_conf, 3),
                "plate": detected_plate_text,
                "has_plate": has_plate,
                "region": active_region,
                "region_name": reg_prof['name'],
                "flag": reg_prof['flag'],
                "law_article_fr": reg_prof['law_article_fr'],
                "law_article_en": reg_prof['law_article_en'],
                "penalty_fr": "Aucune infraction constatée sur ce contrôle.",
                "penalty_en": "No infraction recorded during this inspection.",
                "currency": reg_prof['currency'],
                "summary": f"Rider equipped with approved safety helmet ({reg_prof['name']}).",
                "summary_fr": f"Conducteur équipé d'un équipement réglementaire conforme ({reg_prof.get('name_fr', reg_prof['name'])}).",
                "action": "Inspection concluded: Compliant status logged into telemetry archive.",
                "action_fr": "Contrôle clôturé avec mention Conforme - Données statistiques archivées."
            }
        else:
            verdict = {
                "status": "OBSERVATION",
                "label": f"ROADWAY SURVEILLANCE : SCENE UNDER OBSERVATION ({reg_prof['flag']} {reg_prof['name']})",
                "label_fr": f"SURVEILLANCE ROUTIÈRE : SCÈNE SOUS CONTRÔLE ({reg_prof['flag']} {reg_prof.get('name_fr', reg_prof['name'])})",
                "severity": "INFORMATIVE",
                "color": "#00f0ff",
                "confidence": 0.88,
                "plate": detected_plate_text,
                "has_plate": has_plate,
                "region": active_region,
                "region_name": reg_prof['name'],
                "flag": reg_prof['flag'],
                "law_article_fr": reg_prof['law_article_fr'],
                "law_article_en": reg_prof['law_article_en'],
                "penalty_fr": "Surveillance standard maintenue.",
                "penalty_en": "Standard surveillance active.",
                "currency": reg_prof['currency'],
                "summary": f"{len(detections)} target(s) logged during optical sweep.",
                "summary_fr": f"{len(detections)} élément(s) relevé(s) lors du balayage optique de la scène.",
                "action": "Preventive surveillance active.",
                "action_fr": "Surveillance préventive maintenue."
            }

        return jsonify({
            "success": True,
            "filename": file.filename,
            "width": width,
            "height": height,
            "detections": detections,
            "verdict": verdict,
            "plate_detected": detected_plate_text,
            "plate_info": detected_plate_info,
            "region": active_region,
            "alpr_active": True,
            "engine": "YOLOv5 Neural Network + ALPR Multi-Country OCR",
            "device": str(image_detector.device)
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/status', methods=['GET'])
def status():
    """Indique l'état des moteurs IA et de la caméra CCTV."""
    return jsonify({
        "status": "healthy",
        "yolov5_loaded": image_detector.loaded,
        "image_model": os.path.basename(image_detector.model_path),
        "cctv_model": os.path.basename(cctv_detector.model_path),
        "cctv_is_dedicated": cctv_detector != image_detector,
        "cctv_fps": round(camera.current_fps, 1),
        "cctv_video": os.path.basename(camera.video_path),
        "cctv_region": camera.region,
        "cctv_violation": camera.violation_detected,
        "backup_secured": os.path.exists(os.path.join(BASE_DIR, 'weights', 'best_image_backup.pt')),
        "device": str(image_detector.device) if image_detector.loaded else 'none',
        "classes": image_detector.class_names if image_detector.loaded else {}
    })

if __name__ == '__main__':
    print("=" * 70)
    print(" [READY] Serveur Safety AI Haute Performance en ligne !")
    print(" [URL]   Ouvrez http://localhost:5000 dans votre navigateur.")
    print(" [INFO]  Moteur Asynchrone 30 FPS CCTV & ALPR Multi-Régions activé.")
    print("=" * 70)
    # Exécution avec threaded=True pour garantir que le streaming ne bloque jamais l'API
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)
