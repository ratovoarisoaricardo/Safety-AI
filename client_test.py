import os
import sys
import json

# Try importing requests
try:
    import requests
except ImportError:
    print(" [WARNING] The 'requests' library is not installed.")
    print("           Please install it using: pip install requests")
    sys.exit(1)

# API endpoint details
PORT = 5000
URL = f"http://localhost:{PORT}/api/detect"

def test_api():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to sample image
    img_path = os.path.join(base_dir, 'dataset', 'images', 'train', 'image1.jpg')
    
    # Check if mock images exist. If not, generate them.
    if not os.path.exists(img_path):
        print(" [INFO] Mock images not found. Attempting to generate them...")
        try:
            from dataset.generate_mock_data import create_mock_dataset
            create_mock_dataset()
        except ImportError:
            # Fallback path if directory import structure is different
            sys.path.append(os.path.join(base_dir, 'dataset'))
            try:
                from generate_mock_data import create_mock_dataset
                create_mock_dataset()
            except Exception as e:
                print(f" [ERROR] Could not generate mock images: {e}")
                sys.exit(1)

    if not os.path.exists(img_path):
        print(f" [ERROR] Sample test image not found at: {img_path}")
        print("          Please run: python dataset/generate_mock_data.py first.")
        sys.exit(1)

    print(f" [INFO] Preparing to send image: {os.path.basename(img_path)}")
    print(f" [INFO] Target endpoint: {URL}")

    # Prepare file payload
    try:
        with open(img_path, 'rb') as img_file:
            files = {'image': (os.path.basename(img_path), img_file, 'image/jpeg')}
            
            print(" [INFO] Sending POST request...")
            response = requests.post(URL, files=files)
            
            # Print status
            print(f" [STATUS] HTTP Response Code: {response.status_code}")
            
            if response.status_code == 200:
                result_json = response.json()
                print("\n [SUCCESS] API Response Payload:")
                print(json.dumps(result_json, indent=4))
                
                # Check for detections
                detections = result_json.get("detections", [])
                print(f"\n [SUMMARY] Found {len(detections)} compliance targets:")
                for i, det in enumerate(detections, 1):
                    print(f"   {i}. Class: {det['class']} (Confidence: {det['confidence']*100:.0f}%) | Box: {det['box']}")
            else:
                print(" [ERROR] Server returned error details:")
                print(response.text)
                
    except requests.exceptions.ConnectionError:
        print(f"\n [ERROR] Connection refused at {URL}")
        print(f"         Please start the Flask API server first by running: python server.py")
        sys.exit(1)
    except Exception as e:
        print(f" [ERROR] Unexpected exception occurred: {e}")
        sys.exit(1)

if __name__ == '__main__':
    test_api()
