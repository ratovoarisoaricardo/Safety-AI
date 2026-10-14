import os
import shutil
import cv2

def generate_samples():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(base_dir))
    os.makedirs(base_dir, exist_ok=True)
    
    brain_dir = r"C:\Users\ABCD\.gemini\antigravity-ide\brain\c3a628c0-fcad-435e-81b4-aa4b369327ea"
    
    # 1. Base 6 scenario images from original archive
    uploaded_files = [
        "media__1783427075665.png", # 1: girl rider
        "media__1783427256585.png", # 2: motorcycle KTM road
        "media__1783427335917.png", # 3: criminology rider
        "media__1783427352169.png", # 4: double rider front
        "media__1783427368847.png", # 5: double rider front lane
        "media__1783427409106.png"  # 6: scooter rider front intersection
    ]
    
    if os.path.exists(brain_dir):
        for idx, fname in enumerate(uploaded_files, 1):
            src_path = os.path.join(brain_dir, fname)
            dst_path = os.path.join(base_dir, f"sample_{idx:02d}.png")
            if os.path.exists(src_path) and not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)

    # 2. Extract scenarios 7 through 12 from authentic project videos if missing
    additional_targets = [
        ('sample_07.png', 'output_violation_demo.mp4', 50),
        ('sample_08.png', 'saigon_part.webm', 80),
        ('sample_09.png', 'output_violation_demo.mp4', 120),
        ('sample_10.png', 'output_violation_demo.mp4', 200),
        ('sample_11.png', 'saigon_part.webm', 150),
        ('sample_12.png', 'saigon_part.webm', 220),
    ]

    for out_name, vid_name, frame_no in additional_targets:
        dst = os.path.join(base_dir, out_name)
        if not os.path.exists(dst):
            vid_path = os.path.join(project_root, vid_name)
            if os.path.exists(vid_path):
                cap = cv2.VideoCapture(vid_path)
                count = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    count += 1
                    if count == frame_no:
                        cv2.imwrite(dst, frame)
                        break
                cap.release()

    existing = [f for f in sorted(os.listdir(base_dir)) if f.startswith("sample_") and f.endswith(".png")]
    print(f"[INFO] Found {len(existing)} sample scenarios ready on disk in {base_dir}")

if __name__ == '__main__':
    generate_samples()
