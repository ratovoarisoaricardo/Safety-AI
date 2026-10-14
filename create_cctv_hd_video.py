import cv2
import numpy as np
import os
import math
from PIL import Image

def build_cctv_hd_video(output_path='cctv_traffic_hd.mp4', width=1280, height=720, fps=30):
    print(f" [INFO] Génération de la nouvelle vidéo CCTV Haute Définition : {output_path}...")
    
    # Charger les clichés réels sources contenant les plaques nettes et motards
    base_dir = os.path.dirname(os.path.abspath(__file__))
    s4_path = os.path.join(base_dir, 'static', 'samples', 'sample_04.png')
    s1_path = os.path.join(base_dir, 'static', 'samples', 'sample_01.png')
    s5_path = os.path.join(base_dir, 'static', 'samples', 'sample_05.png')
    
    img_s4 = cv2.imread(s4_path) if os.path.exists(s4_path) else None
    img_s1 = cv2.imread(s1_path) if os.path.exists(s1_path) else None
    img_s5 = cv2.imread(s5_path) if os.path.exists(s5_path) else None

    # Codec MP4 compatible navigateur
    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    except Exception:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        raise RuntimeError("Impossible d'initialiser le VideoWriter OpenCV")

    # Séquences composées de 30 secondes au total (900 frames à 30 FPS)
    # Séquence 1 (0-10s / 300 frames) : Motard SANS CASQUE avec plaque TN 09 BJ 4054 nette qui avance vers la caméra
    # Séquence 2 (10-20s / 300 frames) : Motard AVEC CASQUE conforme
    # Séquence 3 (20-30s / 300 frames) : Carrefour urbain avec plusieurs usagers et plaques nettes (TS 09 EA 1252)

    total_frames = 30 * fps  # 900 frames

    def create_scene_frame(source_bgr, progress, location_str, violation_flag):
        # Créer le fond urbain CCTV 720p
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Dégradé d'asphalte et chaussée
        for y in range(height):
            shade = int(28 + 18 * (y / height))
            canvas[y, :] = (shade, shade, shade + 4)
            
        # Lignes de voies routières en perspective
        cv2.line(canvas, (int(width * 0.42), 0), (int(width * 0.15), height), (80, 80, 90), 3)
        cv2.line(canvas, (int(width * 0.58), 0), (int(width * 0.85), height), (80, 80, 90), 3)
        
        # Marquages au sol pointillés
        for my in range(120, height, 80):
            cv2.line(canvas, (int(width * 0.50), my), (int(width * 0.50), min(height, my + 45)), (180, 180, 180), 4)

        # Simulation de l'approche du véhicule (effet zoom progressif)
        # progress va de 0.0 à 1.0
        sh, sw = source_bgr.shape[:2]
        # Échelle : commence à 0.55 et monte jusqu'à 1.05
        scale = 0.55 + 0.48 * progress
        scaled_w = int(sw * scale)
        scaled_h = int(sh * scale)
        
        scaled = cv2.resize(source_bgr, (scaled_w, scaled_h), interpolation=cv2.INTER_CUBIC)
        
        # Position du motard qui descend la voie centrale vers la caméra
        pos_y = int((height - scaled_h) * 0.45 + (height - scaled_h) * 0.45 * progress)
        pos_x = int((width - scaled_w) * 0.50 + math.sin(progress * math.pi) * 40)
        
        # Incrustation douce
        x1 = max(0, pos_x)
        y1 = max(0, pos_y)
        x2 = min(width, pos_x + scaled_w)
        y2 = min(height, pos_y + scaled_h)
        
        sx1 = max(0, -pos_x)
        sy1 = max(0, -pos_y)
        sx2 = sx1 + (x2 - x1)
        sy2 = sy1 + (y2 - y1)
        
        if x2 > x1 and y2 > y1 and sx2 > sx1 and sy2 > sy1:
            canvas[y1:y2, x1:x2] = scaled[sy1:sy2, sx1:sx2]

        # Grain CCTV cybernétique très léger pour réalisme
        noise = np.random.randint(-5, 5, (height, width, 3), dtype=np.int16)
        canvas_with_noise = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Incrustation HUD de Télé-Surveillance Professionnelle
        # Réticules dans les coins
        corner_len = 35
        color_hud = (0, 240, 255)
        # Haut-Gauche
        cv2.line(canvas_with_noise, (25, 25), (25 + corner_len, 25), color_hud, 2)
        cv2.line(canvas_with_noise, (25, 25), (25, 25 + corner_len), color_hud, 2)
        # Haut-Droite
        cv2.line(canvas_with_noise, (width - 25, 25), (width - 25 - corner_len, 25), color_hud, 2)
        cv2.line(canvas_with_noise, (width - 25, 25), (width - 25, 25 + corner_len), color_hud, 2)
        # Bas-Gauche
        cv2.line(canvas_with_noise, (25, height - 25), (25 + corner_len, height - 25), color_hud, 2)
        cv2.line(canvas_with_noise, (25, height - 25), (25, height - 25 - corner_len), color_hud, 2)
        # Bas-Droite
        cv2.line(canvas_with_noise, (width - 25, height - 25), (width - 25 - corner_len, height - 25), color_hud, 2)
        cv2.line(canvas_with_noise, (width - 25, height - 25), (width - 25, height - 25 - corner_len), color_hud, 2)

        # Horodatage & Métadonnées
        time_sec = int(progress * 10)
        cv2.putText(canvas_with_noise, f"REC [LIVE 30FPS] - {location_str}", (35, 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(canvas_with_noise, f"OPTICAL TARGETING HD | 1280x720 ALPR CALIBRATED", (35, 75), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 240, 255), 1)

        return canvas_with_noise

    print(" [INFO] Synthèse des trames vidéo...")
    for f in range(total_frames):
        sec_idx = f // (10 * fps)
        sub_progress = (f % (10 * fps)) / float(10 * fps)

        if sec_idx == 0 and img_s4 is not None:
            # Séquence 1 : Infraction sans casque (Scooter bleu avec plaque TN 09 BJ 4054 nette)
            frame = create_scene_frame(img_s4, sub_progress, "CAM-02 INTERSECTION NORD [SANS CASQUE]", True)
        elif sec_idx == 1 and img_s1 is not None:
            # Séquence 2 : Conforme avec casque
            frame = create_scene_frame(img_s1, sub_progress, "CAM-02 SECTEUR EST [CASQUE CONFORME]", False)
        elif img_s5 is not None:
            # Séquence 3 : Groupe circulation avec plaques nettes (TS 09 EA 1252)
            frame = create_scene_frame(img_s5, sub_progress, "CAM-02 AVENUE PRINCIPALE [DÉTECTION MULTI-CIBLES]", True)
        else:
            frame = create_scene_frame(img_s4, sub_progress, "CAM-02 SURVEILLANCE ROUTIÈRE", True)

        out.write(frame)

    out.release()
    file_size = os.path.getsize(output_path)
    print(f" [SUCCESS] Nouvelle vidéo CCTV générée avec succès : {output_path} ({file_size:,} octets, {total_frames} frames) !")

if __name__ == '__main__':
    build_cctv_hd_video()
