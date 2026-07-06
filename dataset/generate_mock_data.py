import os
import random
from PIL import Image, ImageDraw

def create_mock_dataset():
    # Define directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    splits = {
        'train': ['image1.jpg', 'image2.jpg', 'image3.jpg'],
        'val': ['image4.jpg', 'image5.jpg']
    }
    
    # 0: Helmet, 1: No_Helmet, 2: License_Plate, 3: Seatbelt, 4: No_Seatbelt
    classes = {
        'Helmet': 0,
        'No_Helmet': 1,
        'License_Plate': 2,
        'Seatbelt': 3,
        'No_Seatbelt': 4
    }

    print("Generating mock dataset files with 5 classes...")

    for split, files in splits.items():
        img_dir = os.path.join(base_dir, 'images', split)
        lbl_dir = os.path.join(base_dir, 'labels', split)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        for filename in files:
            # Create a 640x640 mock street image (gray background)
            img = Image.new('RGB', (640, 640), color=(120, 120, 120))
            draw = ImageDraw.Draw(img)

            # Draw a simulated road
            draw.polygon([(100, 640), (280, 200), (360, 200), (540, 640)], fill=(70, 70, 70))
            # Draw lane lines
            draw.line([(320, 200), (320, 640)], fill=(255, 255, 255), width=4)

            labels = []

            # 1. Helmet (class 0) or No_Helmet (class 1)
            # Motorbike rider simulation
            if random.random() > 0.2:
                # Motorbike rider head position
                x0, y0 = 250, 280
                x1, y1 = 290, 320
                
                is_compliant = random.random() > 0.4
                if is_compliant:
                    # Draw Helmet (Green circle)
                    draw.ellipse([x0, y0, x1, y1], fill=(16, 185, 129), outline=(10, 110, 80), width=2)
                    c_name = 'Helmet'
                else:
                    # Draw No_Helmet (Red circle/face)
                    draw.ellipse([x0, y0, x1, y1], fill=(239, 68, 68), outline=(150, 20, 20), width=2)
                    # Add skin tone center
                    draw.ellipse([x0+4, y0+4, x1-4, y1-4], fill=(251, 207, 232))
                    c_name = 'No_Helmet'
                
                # YOLO format: class x_center y_center width height (normalized)
                x_center = (x0 + x1) / 2.0 / 640.0
                y_center = (y0 + y1) / 2.0 / 640.0
                w = (x1 - x0) / 640.0
                h = (y1 - y0) / 640.0
                labels.append(f"{classes[c_name]} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

            # 2. License_Plate (class 2)
            # Car plate position
            if random.random() > 0.2:
                x0, y0 = 280, 480
                x1, y1 = 360, 510
                
                # Draw Car License Plate (white rectangle)
                draw.rectangle([x0, y0, x1, y1], fill=(255, 255, 255), outline=(0, 240, 255), width=2)
                # Draw license text simulator
                draw.text((x0+12, y0+8), "SF-96-AI", fill=(0, 0, 0))
                
                x_center = (x0 + x1) / 2.0 / 640.0
                y_center = (y0 + y1) / 2.0 / 640.0
                w = (x1 - x0) / 640.0
                h = (y1 - y0) / 640.0
                labels.append(f"{classes['License_Plate']} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

            # 3. Seatbelt (class 3) or No_Seatbelt (class 4)
            # Front windshield driver area simulation
            if random.random() > 0.3:
                # Driver torso position in windshield
                x0, y0 = 310, 400
                x1, y1 = 360, 460
                
                # Draw driver silhouette
                draw.ellipse([x0+5, y0, x1-5, y0+20], fill=(200, 200, 200)) # head
                draw.polygon([(x0, y1), (x1, y1), (x1-5, y0+20), (x0+5, y0+20)], fill=(100, 100, 100)) # torso
                
                is_belted = random.random() > 0.5
                if is_belted:
                    # Draw Seatbelt (green diagonal belt)
                    draw.line([x0+5, y0+20, x1-5, y1], fill=(16, 185, 129), width=4)
                    c_name = 'Seatbelt'
                else:
                    # Draw No_Seatbelt (red crossed indicator)
                    draw.line([x0+5, y0+20, x1-5, y1], fill=(239, 68, 68), width=1)
                    # Red cross indicator
                    draw.line([x0+15, y0+25, x1-15, y0+45], fill=(239, 68, 68), width=3)
                    draw.line([x1-15, y0+25, x0+15, y0+45], fill=(239, 68, 68), width=3)
                    c_name = 'No_Seatbelt'
                
                x_center = (x0 + x1) / 2.0 / 640.0
                y_center = (y0 + y1) / 2.0 / 640.0
                w = (x1 - x0) / 640.0
                h = (y1 - y0) / 640.0
                labels.append(f"{classes[c_name]} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

            # Save image
            img_path = os.path.join(img_dir, filename)
            img.save(img_path)

            # Save label file
            lbl_filename = filename.replace('.jpg', '.txt')
            lbl_path = os.path.join(lbl_dir, lbl_filename)
            with open(lbl_path, 'w') as f:
                f.write('\n'.join(labels))

            print(f" Saved split '{split}': {img_path} and {lbl_path}")

    print("Mock dataset generation completed successfully!")

if __name__ == '__main__':
    create_mock_dataset()
