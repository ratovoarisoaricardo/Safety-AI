# SafeCityAI: Traffic Violation Detection Console

An automated, real-time traffic safety compliance monitoring dashboard powered by a custom **YOLOv5** object detection model.

This project is submitted as a **Minor Project** for Computer Vision Engineering.

---

## 🚀 Getting Started

The project is designed to run out of the box. Follow these steps to set up the environment and launch the server.

### Method 1: Using the Automated Startup Script (Windows)
1. Navigate to the root directory of the project.
2. Double-click the **`run_safecity.bat`** file.
3. The script will automatically:
   * Verify Python installation.
   * Upgrade `pip` and install all required libraries from `requirements.txt`.
   * Compile and generate the H.264 browser-compatible demonstration video (`output_violation_demo.mp4`).
   * Launch the Flask server on port `5000`.
4. Once completed, open your web browser and navigate to: **`http://localhost:5000`**

### Method 2: Manual Setup via Terminal
If you prefer running commands manually:
```bash
# 1. Navigate to the project directory
cd safecity-ai

# 2. Activate your virtual environment (if applicable)
# For Windows:
..\venv\Scripts\activate

# 3. Install required libraries
pip install -r requirements.txt
pip install torch torchvision ultralytics seaborn scipy

# 4. Generate the demo video
python demo_video.py

# 5. Start the Flask application
python server.py
```
Open your browser and navigate to: **`http://localhost:5000`**

---

## 🔍 Features to Test

Once the web interface is open, you can test the following features:

### 1. Real-Time Image Detection
* On the left panel ("Traffic Scene Image Analyzer"), click the upload area or drag and drop a road scene photo (e.g., from `dataset/images/train/image1.jpg`).
* The Flask server will run the custom YOLOv5 model (`best.pt`) on the image and return a bounding box overlay highlighting compliance targets:
  * **Helmet** (Green Box)
  * **No_Helmet** (Red Box - Violation)
  * **License_Plate** (Cyan Box)
  * **Seatbelt** (Green Box)
  * **No_Seatbelt** (Red Box - Violation)

### 2. Live CCTV Feed Simulation
* The right panel shows an interactive intersection surveillance simulation rendering the HUD detection layout at 30 FPS.
* Check the blinking alert notifications and the system performance metrics.

### 3. API Response Inspector
* View the raw, parsed JSON payload returned by the `/api/detect` endpoint in the bottom-right inspector window.

### 4. REST API Endpoint Validation
* Open a new terminal and execute the test client script:
  ```bash
  python client_test.py
  ```
  It validates the communication with the Flask backend, uploads a sample image, and prints the returned JSON coordinates.

---

## 📁 Repository Structure

* [server.py](file:///c:/Users/ABCD/OneDrive/Documents/AI/safecity-ai/server.py): Flask application file hosting REST API endpoints and web routing.
* [demo_video.py](file:///c:/Users/ABCD/OneDrive/Documents/AI/safecity-ai/demo_video.py): Video frame processing script using OpenCV and custom weights.
* [client_test.py](file:///c:/Users/ABCD/OneDrive/Documents/AI/safecity-ai/client_test.py): Local API client validation script.
* [best.pt](file:///c:/Users/ABCD/OneDrive/Documents/AI/safecity-ai/best.pt): Fine-tuned PyTorch custom weights for YOLOv5.
* [dataset/](file:///c:/Users/ABCD/OneDrive/Documents/AI/safecity-ai/dataset): Mock training/validation street datasets and YOLO mappings.
* [notebooks/SafeCityAI_YOLOv5.ipynb](file:///c:/Users/ABCD/OneDrive/Documents/AI/safecity-ai/notebooks/SafeCityAI_YOLOv5.ipynb): Google Colab pipeline notebook.
* [templates/](file:///c:/Users/ABCD/OneDrive/Documents/AI/safecity-ai/templates) & [static/](file:///c:/Users/ABCD/OneDrive/Documents/AI/safecity-ai/static): Dashboard console user interface files (HTML, CSS, JS).
