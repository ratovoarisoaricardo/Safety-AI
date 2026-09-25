#!/usr/bin/env bash
set -e
echo "=== SafeCityAI (Linux / macOS) ==="
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
if [ ! -f "output_violation_demo.mp4" ]; then
    python demo_video.py
fi
echo "Starting Flask Server on http://localhost:5000..."
python server.py