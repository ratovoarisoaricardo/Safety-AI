# ==============================================================================
# SafeCityAI - CCTV Video Frame Extractor & Dataset Builder
# Extrait des frames depuis des vidéos CCTV pour alimenter datasets/dataset_cctv/
# ==============================================================================

import os
import cv2
import argparse
import random

def extract_frames(video_path, output_dir="datasets/dataset_cctv", interval=10, val_ratio=0.2):
    if not os.path.exists(video_path):
        print(f" [ERREUR] Fichier vidéo introuvable : {video_path}")
        return

    train_img_dir = os.path.join(output_dir, "images", "train")
    val_img_dir = os.path.join(output_dir, "images", "val")
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    base_name = os.path.splitext(os.path.basename(video_path))[0]

    print(f" [INFO] Ouverture de {video_path} : {total_frames} frames ({fps:.1f} FPS)")
    print(f" [INFO] Extraction : 1 frame toutes les {interval} frames...")

    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % interval == 0:
            is_val = random.random() < val_ratio
            dest_dir = val_img_dir if is_val else train_img_dir
            split_name = "val" if is_val else "train"
            
            img_filename = f"{base_name}_frame_{frame_idx:06d}.jpg"
            img_filepath = os.path.join(dest_dir, img_filename)
            
            cv2.imwrite(img_filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_count += 1

        frame_idx += 1

    cap.release()
    print(f" [SUCCÈS] {saved_count} frames extraites et enregistrées dans {output_dir}/images/ (train & val) !")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extraire des frames CCTV pour la base de données")
    parser.add_argument("--video", type=str, default="input_traffic.mp4", help="Chemin du fichier vidéo")
    parser.add_argument("--interval", type=int, default=15, help="Extraire 1 image toutes les N frames")
    args = parser.parse_args()

    extract_frames(args.video, interval=args.interval)
