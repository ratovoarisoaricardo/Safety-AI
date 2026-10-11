# ==============================================================================
# SafeCityAI - Automatic YOLO Label Generator for CCTV Dataset
# Utilise le modèle neural pour pré-annoter les frames extraites de la CCTV
# ==============================================================================

import os
from PIL import Image
from model_engine import detector

CLASS_MAP = {
    'License_Plate': 0,
    'number plate': 0,
    'Rider': 1,
    'rider': 1,
    'Helmet': 2,
    'with helmet': 2,
    'with_helmet': 2,
    'No_Helmet': 3,
    'without helmet': 3,
    'without_helmet': 3
}

def annotate_folder(img_folder, label_folder):
    os.makedirs(label_folder, exist_ok=True)
    images = [f for f in os.listdir(img_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f" [INFO] Annotation de {len(images)} images dans {img_folder}...")

    total_annotations = 0
    for img_name in images:
        img_path = os.path.join(img_folder, img_name)
        base_name = os.path.splitext(img_name)[0]
        label_path = os.path.join(label_folder, f"{base_name}.txt")

        try:
            pil_img = Image.open(img_path).convert('RGB')
            w, h = pil_img.size
            detections = detector.detect_image(pil_img, conf_thres=0.20)

            lines = []
            for det in detections:
                cls_name = det['class']
                if cls_name not in CLASS_MAP:
                    continue
                cls_id = CLASS_MAP[cls_name]
                bx, by, bw, bh = det['box']

                # Normalisation YOLO (cx, cy, w, h)
                cx = (bx + bw / 2.0) / w
                cy = (by + bh / 2.0) / h
                nw = bw / w
                nh = bh / h

                lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                total_annotations += 1

            with open(label_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(lines) + "\n")

        except Exception as e:
            print(f" [ERREUR] {img_name}: {e}")

    print(f" [SUCCÈS] {len(images)} fichiers .txt générés ({total_annotations} objets détectés et annotés) !")

if __name__ == '__main__':
    cctv_root = os.path.join(os.path.dirname(__file__), 'datasets', 'dataset_cctv')
    annotate_folder(os.path.join(cctv_root, 'images', 'train'), os.path.join(cctv_root, 'labels', 'train'))
    annotate_folder(os.path.join(cctv_root, 'images', 'val'), os.path.join(cctv_root, 'labels', 'val'))
