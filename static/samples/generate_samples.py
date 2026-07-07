import os
import random
from PIL import Image, ImageDraw

def generate_samples():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(base_dir, exist_ok=True)
    
    print("Generating 30 diverse mock street scenes for the test gallery...")
    
    # 0: Helmet, 1: No_Helmet, 2: License_Plate, 3: Seatbelt, 4: No_Seatbelt
    for i in range(1, 31):
        # Create a 640x640 mock street image with varying background lighting
        brightness = random.randint(80, 160)
        img = Image.new('RGB', (640, 640), color=(brightness, brightness, brightness))
        draw = ImageDraw.Draw(img)
        
        # Vary road color and position slightly
        road_left = random.randint(60, 120)
        road_right = random.randint(500, 580)
        draw.polygon([(road_left, 640), (270, 220), (370, 220), (road_right, 640)], fill=(50, 50, 50))
        
        # Lane divider dashes
        draw.line([(320, 220), (320, 640)], fill=(255, 255, 255), width=3)
        
        # Add side foliage blocks (green rectangles)
        draw.rectangle([0, 0, 80, 640], fill=(20, 50, 20))
        draw.rectangle([560, 0, 640, 640], fill=(20, 50, 20))
        
        filename = f"sample_{i:02d}.jpg"
        
        # Decides scenario based on index:
        # Motorcycle scenes (odd indices), Car scenes (even indices)
        if i % 2 == 1:
            # Motorcycle rider
            x0 = random.randint(220, 280)
            y0 = random.randint(250, 310)
            x1, y1 = x0 + 40, y0 + 40
            
            # Helmet compliance (2 out of 3 times compliant)
            is_helmet = (i % 3 != 1)
            if is_helmet:
                # Green helmet circle
                draw.ellipse([x0, y0, x1, y1], fill=(16, 185, 129), outline=(10, 110, 80), width=2)
                # Visor details
                draw.rectangle([x0+15, y0+10, x1, y0+18], fill=(0, 0, 0))
            else:
                # Red face circle (No Helmet)
                draw.ellipse([x0, y0, x1, y1], fill=(239, 68, 68), outline=(150, 20, 20), width=2)
                draw.ellipse([x0+4, y0+4, x1-4, y1-4], fill=(251, 207, 232))
                
            # Add license plate
            px0, py0 = x0 + 10, y0 + 120
            px1, py1 = px0 + 45, py0 + 16
            draw.rectangle([px0, py0, px1, py1], fill=(255, 255, 255), outline=(0, 240, 255), width=1)
            draw.text((px0+6, py0+3), f"MC-{i:02d}", fill=(0, 0, 0))
            
        else:
            # Car windshield scene
            cx0 = random.randint(280, 320)
            cy0 = random.randint(380, 420)
            cx1, cy1 = cx0 + 50, cy0 + 60
            
            # Windshield box
            draw.rectangle([cx0-20, cy0-40, cx1+20, cy1], fill=(80, 100, 120), outline=(200, 200, 200), width=2)
            
            # Driver silhouette
            draw.ellipse([cx0+5, cy0-25, cx1-5, cy0], fill=(180, 180, 180)) # head
            draw.polygon([(cx0, cy1), (cx1, cy1), (cx1-5, cy0), (cx0+5, cy0)], fill=(80, 80, 80)) # torso
            
            # Seatbelt compliance
            is_seatbelt = (i % 4 != 0)
            if is_seatbelt:
                # Green diagonal line
                draw.line([cx0+5, cy0, cx1-5, cy1], fill=(16, 185, 129), width=4)
            else:
                # Red cross on chest
                draw.line([cx0+15, cy0+10, cx1-15, cy0+30], fill=(239, 68, 68), width=3)
                draw.line([cx1-15, cy0+10, cx0+15, cy0+30], fill=(239, 68, 68), width=3)
                
            # License plate on bumper
            px0, py0 = cx0 - 15, cy0 + 80
            px1, py1 = px0 + 80, py0 + 20
            draw.rectangle([px0, py0, px1, py1], fill=(255, 255, 255), outline=(0, 240, 255), width=2)
            draw.text((px0+12, py0+4), f"CAR-SF-{i:02d}", fill=(0, 0, 0))
            
        # Draw some indicator index text on top left of image for gallery clarity
        draw.rectangle([5, 5, 110, 22], fill=(15, 23, 42))
        draw.text((10, 8), f"Test Scenario {i:02d}", fill=(0, 240, 255))
        
        file_path = os.path.join(base_dir, filename)
        img.save(file_path)

    print("Successfully generated 30 sample gallery images!")

if __name__ == '__main__':
    generate_samples()
