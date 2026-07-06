import os
import sys

# Try importing cv2 and numpy
try:
    import cv2
    import numpy as np
except ImportError:
    print(" [WARNING] OpenCV ('opencv-python') or 'numpy' is not installed.")
    print("           Please install them: pip install opencv-python numpy")
    sys.exit(1)

# Check for PyTorch/YOLOv5
MODEL_LOADED = False
model = None
MODEL_PATH = 'best.pt'

try:
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if os.path.exists(MODEL_PATH):
        print(f" [INFO] Custom weights found. Loading Custom YOLOv5 from '{MODEL_PATH}'...")
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, device=device)
        MODEL_LOADED = True
    else:
        print(" [INFO] 'best.pt' not found. Running demo stream using simulation renderer...")
except Exception as e:
    print(f" [INFO] PyTorch not initialized: {e}. Running in simulation mode.")

def create_synthetic_traffic_video(filename, width=640, height=480, duration_sec=10, fps=30):
    """Generates a synthetic traffic video frame-by-frame for testing."""
    print(f" [INFO] Generating synthetic traffic clip: {filename} ({duration_sec}s, {fps}fps)...")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    total_frames = duration_sec * fps
    
    for frame_idx in range(total_frames):
        # Base frame (gray road and background)
        frame = np.ones((height, width, 3), dtype=np.uint8) * 40 # Dark gray background
        
        # Road lane polygon
        road_pts = np.array([[80, height], [260, 150], [380, 150], [560, height]], np.int32)
        cv2.fillPoly(frame, [road_pts], (70, 70, 70))
        
        # Lanes dash
        cv2.line(frame, (320, 150), (320, height), (255, 255, 255), thickness=2)
        
        # Draw some trees/buildings simple representation
        cv2.rectangle(frame, (0, 0), (70, height), (30, 60, 30), -1)
        cv2.rectangle(frame, (width - 70, 0), (width, height), (30, 60, 30), -1)
        
        # Simulated objects coordinates
        # 1. Car driving down road (coming closer, scaling up)
        car_progress = (frame_idx % 150) / 150.0
        car_w = int(40 + car_progress * 100)
        car_h = int(25 + car_progress * 60)
        car_y = int(160 + car_progress * 260)
        car_x = int(370 + car_progress * 70)
        
        # Draw Car (Blue block)
        cv2.rectangle(frame, (car_x - car_w//2, car_y - car_h), (car_x + car_w//2, car_y), (180, 100, 30), -1)
        # Windows
        cv2.rectangle(frame, (car_x - car_w//3, car_y - car_h + 5), (car_x + car_w//3, car_y - car_h + 15), (255, 240, 200), -1)
        
        # Driver silhouette in car window
        driver_x = int(car_x - car_w//6)
        driver_y = int(car_y - car_h//1.5)
        driver_r = int(5 * (0.5 + car_progress * 0.8))
        cv2.circle(frame, (driver_x, driver_y), driver_r, (150, 150, 150), -1)
        cv2.rectangle(frame, (driver_x - driver_r, driver_y + driver_r), (driver_x + driver_r, driver_y + driver_r*2), (100, 100, 100), -1)
        
        # License Plate (white rect)
        plate_w = int(15 + car_progress * 20)
        plate_h = int(6 + car_progress * 8)
        plate_x = car_x - plate_w//2
        plate_y = car_y - plate_h - 4
        cv2.rectangle(frame, (plate_x, plate_y), (plate_x + plate_w, plate_y + plate_h), (255, 255, 255), -1)
        
        # 2. Motorcycle Rider (going left to right, constant speed, violation)
        rider_progress = (frame_idx % 200) / 200.0
        rider_x = int(-40 + rider_progress * 720)
        rider_y = int(320 - rider_progress * 100)
        rider_scale = 0.5 + (1.0 - rider_progress) * 0.8
        
        # Draw wheels
        cv2.circle(frame, (int(rider_x - 15*rider_scale), rider_y), int(10*rider_scale), (0, 0, 0), -1)
        cv2.circle(frame, (int(rider_x + 15*rider_scale), rider_y), int(10*rider_scale), (0, 0, 0), -1)
        
        # Draw rider body (Red block)
        cv2.rectangle(frame, (int(rider_x - 10*rider_scale), int(rider_y - 45*rider_scale)), 
                      (int(rider_x + 10*rider_scale), rider_y), (50, 50, 220), -1)
        
        # Draw head (pink skin color)
        cv2.circle(frame, (rider_x, int(rider_y - 45*rider_scale)), int(7*rider_scale), (180, 200, 255), -1)
        
        out.write(frame)
        
    out.release()
    print(" [SUCCESS] Synthetic video written.")

def process_video(input_file, output_file):
    if not os.path.exists(input_file):
        create_synthetic_traffic_video(input_file)
        
    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        print(f" [ERROR] Could not open video: {input_file}")
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    print(f" [INFO] Processing {total_frames} frames from {input_file}...")
    print(f" [INFO] Output will save to {output_file}")
    
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Perform detection
        if MODEL_LOADED:
            # PyTorch inference
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(img_rgb)
            predictions = results.pandas().xyxy[0]
            
            for idx, row in predictions.iterrows():
                class_name = row['name']
                conf = float(row['confidence'])
                
                # Check mapping for demo if COCO
                if not os.path.exists(MODEL_PATH):
                    if class_name == 'person':
                        if idx % 3 == 0:
                            class_name = 'Helmet'
                        elif idx % 3 == 1:
                            class_name = 'No_Helmet'
                        else:
                            class_name = 'No_Seatbelt' if idx % 2 == 0 else 'Seatbelt'
                    elif class_name == 'motorcycle':
                        class_name = 'Helmet'
                    elif class_name in ['car', 'truck', 'bus']:
                        class_name = 'License_Plate'
                    else:
                        continue
                
                # Coords
                x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                
                # Color selector
                if class_name in ['Helmet', 'Seatbelt']:
                    color = (0, 255, 0) # Green
                elif class_name in ['No_Helmet', 'No_Seatbelt']:
                    color = (0, 0, 255) # Red
                else:
                    color = (255, 240, 0) # Cyan
                
                # Bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                # Label
                label = f"{class_name} {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            # --- Simulated Render Overlays ---
            # 1. License Plate
            car_progress = (frame_count % 150) / 150.0
            car_w = int(40 + car_progress * 100)
            car_h = int(25 + car_progress * 60)
            car_y = int(160 + car_progress * 260)
            car_x = int(370 + car_progress * 70)
            
            plate_w = int(15 + car_progress * 20)
            plate_h = int(6 + car_progress * 8)
            plate_x = car_x - plate_w//2
            plate_y = car_y - plate_h - 4
            
            # Cyan box
            cv2.rectangle(frame, (plate_x - 3, plate_y - 3), (plate_x + plate_w + 3, plate_y + plate_h + 3), (255, 240, 0), 2)
            cv2.putText(frame, "License_Plate 95%", (plate_x - 3, plate_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 240, 0), 1)
            
            # 2. Driver Seatbelt Compliance (alternates status)
            is_belted = ((frame_count // 150) % 2 == 0)
            driver_x = int(car_x - car_w//6)
            driver_y = int(car_y - car_h//1.5)
            driver_r = int(5 * (0.5 + car_progress * 0.8))
            
            # Bounding box coordinates for seatbelt detection
            belt_x1, belt_y1 = driver_x - driver_r - 2, driver_y - driver_r - 2
            belt_x2, belt_y2 = driver_x + driver_r + 2, driver_y + driver_r*2 + 2
            
            if is_belted:
                cv2.rectangle(frame, (belt_x1, belt_y1), (belt_x2, belt_y2), (0, 255, 0), 2)
                cv2.putText(frame, "Seatbelt 91%", (belt_x1, belt_y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            else:
                cv2.rectangle(frame, (belt_x1, belt_y1), (belt_x2, belt_y2), (0, 0, 255), 2)
                cv2.putText(frame, "No_Seatbelt 88%", (belt_x1, belt_y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
            
            # 3. Motorcycle Rider (No Helmet violation)
            rider_progress = (frame_count % 200) / 200.0
            rider_x = int(-40 + rider_progress * 720)
            rider_y = int(320 - rider_progress * 100)
            rider_scale = 0.5 + (1.0 - rider_progress) * 0.8
            
            head_radius = int(7 * rider_scale)
            head_x = rider_x
            head_y = int(rider_y - 45 * rider_scale)
            
            # Red box
            x1, y1 = head_x - head_radius - 4, head_y - head_radius - 4
            x2, y2 = head_x + head_radius + 4, head_y + head_radius + 4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, "No_Helmet 89%", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
            # Blinking Alert Bar on Screen
            is_helmet_violation = (40 < (frame_count % 200) < 160)
            is_seatbelt_violation = (not is_belted and 30 < (frame_count % 150) < 120)
            
            if is_helmet_violation:
                cv2.rectangle(frame, (10, 10), (width - 10, 45), (0, 0, 255), -1)
                cv2.putText(frame, "TRAFFIC VIOLATION: NO_HELMET DETECTED", (40, 33), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            elif is_seatbelt_violation:
                cv2.rectangle(frame, (10, 10), (width - 10, 45), (0, 0, 255), -1)
                cv2.putText(frame, "TRAFFIC VIOLATION: NO_SEATBELT DETECTED", (40, 33), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            
        # Draw telemetry HUD
        cv2.rectangle(frame, (10, height - 35), (320, height - 10), (0, 0, 0), -1)
        cv2.putText(frame, f"SafeCityAI Live Monitor | Frame: {frame_count}/{total_frames}", 
                    (15, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        out.write(frame)
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"   Processed {frame_count}/{total_frames} frames...")
            
    cap.release()
    out.release()
    print(f" [SUCCESS] Processing complete! Demonstration video saved to: {output_file}")

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_video = os.path.join(base_dir, 'input_traffic.mp4')
    output_video = os.path.join(base_dir, 'output_violation_demo.mp4')
    
    process_video(input_video, output_video)
