# ==============================================================================
# SafeCityAI - CCPD License Plate Dataset Parser & Converter
# Convertisseur officiel du dataset CCPD (https://github.com/detectRecog/CCPD)
# ==============================================================================

import os
import re

# Table de correspondance officielle des caractères chinois et alphanumériques CCPD
PROVINCES = [
    "皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
    "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
    "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁",
    "新", "警", "学", "O"
]

ALPHABETS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
    'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
    'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5',
    '6', '7', '8', '9'
]

def parse_ccpd_filename(filename):
    """
    Décode les métadonnées d'un nom de fichier CCPD :
    Exemple : 025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    parts = base.split('-')
    if len(parts) < 7:
        return None

    # 1. Bounding box [x1, y1, x2, y2]
    bbox_str = parts[2].split('_')
    x1, y1 = [int(v) for v in bbox_str[0].split('&')]
    x2, y2 = [int(v) for v in bbox_str[1].split('&')]
    box = [x1, y1, x2 - x1, y2 - y1]

    # 2. Matricule de la plaque
    plate_indices = [int(idx) for idx in parts[4].split('_')]
    province_idx = plate_indices[0]
    province = PROVINCES[province_idx] if province_idx < len(PROVINCES) else ""
    
    chars = []
    for idx in plate_indices[1:]:
        if idx < len(ALPHABETS):
            chars.append(ALPHABETS[idx])
            
    plate_str = province + "".join(chars)
    plate_ascii = "".join(chars)  # Version ASCII lettres + chiffres

    # 3. Qualité
    brightness = int(parts[5])
    blurriness = int(parts[6])

    return {
        "box": box,
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        "plate_text": plate_str,
        "plate_ascii": plate_ascii,
        "brightness": brightness,
        "blurriness": blurriness
    }

if __name__ == '__main__':
    sample_name = "025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg"
    res = parse_ccpd_filename(sample_name)
    print("Exemple de decodage CCPD :")
    print(f" -> Boite [x, y, w, h] : {res['box']}")
    print(f" -> Matricule ASCII    : {res['plate_ascii']}")
    print(f" -> Luminosite         : {res['brightness']}")
    print(f" -> Flou cinetique     : {res['blurriness']}")
