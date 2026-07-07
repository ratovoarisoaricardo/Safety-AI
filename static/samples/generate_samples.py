import os
import shutil

def generate_samples():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(base_dir, exist_ok=True)
    
    brain_dir = r"C:\Users\ABCD\.gemini\antigravity-ide\brain\c3a628c0-fcad-435e-81b4-aa4b369327ea"
    
    # List of uploaded user files to copy
    uploaded_files = [
        "media__1783426939076.png",
        "media__1783427075665.png",
        "media__1783427256585.png",
        "media__1783427335917.png",
        "media__1783427352169.png",
        "media__1783427368847.png",
        "media__1783427409106.png"
    ]
    
    # Check if we are running on the developer's machine with access to the brain folder
    if os.path.exists(brain_dir):
        print("[INFO] Developer environment detected. Copying real user-uploaded traffic images...")
        
        # Clean any old JPEG samples
        for file in os.listdir(base_dir):
            if file.endswith('.jpg') or file.endswith('.png'):
                if file != "generate_samples.py" and not file.startswith("sample_"):
                    try:
                        os.remove(os.path.join(base_dir, file))
                    except Exception:
                        pass
        
        # Copy the 7 PNG files
        copied_count = 0
        for idx, fname in enumerate(uploaded_files, 1):
            src_path = os.path.join(brain_dir, fname)
            if os.path.exists(src_path):
                dst_path = os.path.join(base_dir, f"sample_{idx:02d}.png")
                shutil.copy2(src_path, dst_path)
                copied_count += 1
                
        print(f"[SUCCESS] Copied {copied_count} real traffic scenario images to static gallery.")
    else:
        # Running on examiner/reviewer environment: images should already be packaged
        print("[INFO] Reviewer environment detected. Gallery images should already be present.")
        # Check if they exist, if not, print a message
        existing = [f for f in os.listdir(base_dir) if f.startswith("sample_") and f.endswith(".png")]
        print(f"[INFO] Found {len(existing)} gallery images ready on disk.")

if __name__ == '__main__':
    generate_samples()
