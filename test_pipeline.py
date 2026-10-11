import os
import cv2
import numpy as np
from PIL import Image
from model_engine import image_detector
from alpr_ocr import alpr_engine

print(f"Image detector loaded: {image_detector.loaded}")

base_samples = "static/samples"

for i in range(1, 7):
    path = f"{base_samples}/sample_{i:02d}.png"
    if not os.path.exists(path):
        continue
    img = Image.open(path).convert("RGB")
    dets = image_detector.detect_image(img)
    
    helmets = [d for d in dets if 'helmet' in d['class'].lower()]
    plates = [d for d in dets if 'plate' in d['class'].lower() or 'number' in d['class'].lower()]
    riders = [d for d in dets if 'rider' in d['class'].lower()]
    
    print(f"\n==================== {path} ====================")
    print(f"Casques: {len(helmets)} | Plaques: {len(plates)} | Motards: {len(riders)}")
    
    # Vérification matricule ALPR pour chaque plaque
    plate_idx = 0
    for d in dets:
        c = d['class']
        conf = d['confidence']
        box = d['box']
        ocr_res = ""
        if 'plate' in c.lower() or 'number' in c.lower():
            bx, by, bw, bh = [int(v) for v in box]
            crop = np.array(img)[by:by+bh, bx:bx+bw]
            if crop.size > 0:
                bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                plate_txt = alpr_engine.read_plate(bgr, plate_index=plate_idx)
                plate_idx += 1
                ocr_res = f" ==> MATRICULE RECHERCHÉ: [{plate_txt}]"
        print(f"  * {c:20s} ({conf:.2f}) {ocr_res}")
