/**
 * SafeCityAI Frontend Controller
 * Manages UI interactions, drag-and-drop file uploads, Flask API calls,
 * bounding box rendering, and the live CCTV traffic inference simulation.
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const detectionViewer = document.getElementById('detectionViewer');
    const sourceImage = document.getElementById('sourceImage');
    const detectionCanvas = document.getElementById('detectionCanvas');
    const resetBtn = document.getElementById('resetBtn');
    const resultsTableContainer = document.getElementById('resultsTableContainer');
    const resultsBody = document.getElementById('resultsBody');
    const jsonResponse = document.getElementById('jsonResponse');
    const totalViolationsCounter = document.getElementById('totalViolationsCounter');
    
    // CCTV Sim Elements
    const cctvSimCanvas = document.getElementById('cctvSimCanvas');
    const cctvTime = document.getElementById('cctvTime');
    const cctvAlert = document.getElementById('cctvAlert');
    const cctvPlayBtn = document.getElementById('cctvPlayBtn');
    
    let isCctvRunning = true;
    let detectionsLogged = 1482; // Initial stats counter
    
    // ----------------------------------------------------
    // 1. Initial Backend Status Check
    // ----------------------------------------------------
    function checkEngineStatus() {
        const statusDot = document.getElementById('statusDot');
        const statusText = document.getElementById('statusText');
        const statusCard = document.getElementById('engineStatusCard');
        
        fetch('/api/status')
            .then(res => res.json())
            .then(data => {
                if (data.yolov5_loaded) {
                    statusDot.className = 'status-dot active';
                    statusText.textContent = data.custom_weights 
                        ? 'Engine Active (YOLOv5 Custom)' 
                        : 'Engine Active (YOLOv5 COCO Demo)';
                    statusDot.classList.add('active');
                } else {
                    statusDot.className = 'status-dot warning';
                    statusText.textContent = 'Simulation Fallback Mode';
                }
            })
            .catch(err => {
                console.log("Server status offline. Running dashboard in simulation mode.");
                statusDot.className = 'status-dot warning';
                statusText.textContent = 'Simulation Mode (Engine Offline)';
            });
    }
    
    checkEngineStatus();
    
    // ----------------------------------------------------
    // 2. Drag & Drop / File Input Handlers
    // ----------------------------------------------------
    
    // Open file selector when clicking the drop zone
    dropZone.addEventListener('click', () => fileInput.click());
    
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleUploadedFile(e.target.files[0]);
        }
    });
    
    // Highlight drop zone on drag hover
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add('dragover');
        }, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('dragover');
        }, false);
    });
    
    // Handle dropped file
    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleUploadedFile(files[0]);
        }
    });
    
    resetBtn.addEventListener('click', resetAnalyzer);
    
    function resetAnalyzer() {
        detectionViewer.classList.add('hidden');
        resultsTableContainer.classList.add('hidden');
        dropZone.classList.remove('hidden');
        if (document.getElementById('gallerySection')) {
            document.getElementById('gallerySection').classList.remove('hidden');
        }
        fileInput.value = '';
        jsonResponse.textContent = JSON.stringify({
            "status": "waiting_for_upload",
            "message": "Upload an image to trigger /api/detect request"
        }, null, 2);
        
        // Clear canvas drawing
        const ctx = detectionCanvas.getContext('2d');
        ctx.clearRect(0, 0, detectionCanvas.width, detectionCanvas.height);
    }
    
    // ----------------------------------------------------
    // 3. Process File and Trigger Inference API
    // ----------------------------------------------------
    function handleUploadedFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please upload a valid image file (JPG, PNG).');
            return;
        }
        
        // Hide drop zone and gallery, show placeholder image to compute coordinates
        dropZone.classList.add('hidden');
        if (document.getElementById('gallerySection')) {
            document.getElementById('gallerySection').classList.add('hidden');
        }
        detectionViewer.classList.remove('hidden');
        
        // Show loading state
        jsonResponse.textContent = "Processing image...\nCalling POST /api/detect endpoint...";
        
        const reader = new FileReader();
        reader.onload = function(event) {
            sourceImage.src = event.target.result;
            sourceImage.onload = function() {
                // Submit image file to API
                uploadImageAndDraw(file);
            };
        };
        reader.readAsDataURL(file);
    }
    
    function uploadImageAndDraw(file) {
        const formData = new FormData();
        formData.append('image', file);
        
        fetch('/api/detect', {
            method: 'POST',
            body: formData
        })
        .then(res => res.json())
        .then(data => {
            // Display JSON output
            jsonResponse.textContent = JSON.stringify(data, null, 2);
            
            if (data.success) {
                renderDetections(data);
            } else {
                alert('Inference error: ' + data.error);
                resetAnalyzer();
            }
        })
        .catch(err => {
            console.error('API call failed:', err);
            // Simulate offline fallback locally
            simulateAPICall(file.name);
        });
    }
    
    // Fallback simulation when Flask server backend isn't responding
    function simulateAPICall(filename) {
        const width = sourceImage.naturalWidth || 640;
        const height = sourceImage.naturalHeight || 640;
        
        const isHelmetDemo = filename.toLowerCase().includes('helmet');
        const isSeatbeltDemo = filename.toLowerCase().includes('seatbelt');
        const detections = [
            {
                "class": isHelmetDemo ? "Helmet" : "No_Helmet",
                "confidence": 0.89,
                "box": [Math.round(width * 0.42), Math.round(height * 0.22), Math.round(width * 0.12), Math.round(height * 0.16)]
            },
            {
                "class": "License_Plate",
                "confidence": 0.94,
                "box": [Math.round(width * 0.38), Math.round(height * 0.65), Math.round(width * 0.20), Math.round(height * 0.08)]
            },
            {
                "class": isSeatbeltDemo ? "Seatbelt" : (Math.random() > 0.5 ? "Seatbelt" : "No_Seatbelt"),
                "confidence": 0.91,
                "box": [Math.round(width * 0.52), Math.round(height * 0.35), Math.round(width * 0.10), Math.round(height * 0.15)]
            }
        ];
        
        const data = {
            "success": true,
            "filename": filename,
            "width": width,
            "height": height,
            "detections": detections,
            "mode": "Client-side simulation fallback (API Offline)"
        };
        
        jsonResponse.textContent = JSON.stringify(data, null, 2);
        renderDetections(data);
    }
    
    // ----------------------------------------------------
    // 4. Render Bounding Boxes on Overlay Canvas
    // ----------------------------------------------------
    function renderDetections(data) {
        const imgWidth = sourceImage.clientWidth;
        const imgHeight = sourceImage.clientHeight;
        
        // Sync canvas display size to matching image size
        detectionCanvas.width = imgWidth;
        detectionCanvas.height = imgHeight;
        
        const ctx = detectionCanvas.getContext('2d');
        ctx.clearRect(0, 0, imgWidth, imgHeight);
        
        // Scale factors from raw image to client viewport image
        const scaleX = imgWidth / data.width;
        const scaleY = imgHeight / data.height;
        
        // Clear results table
        resultsBody.innerHTML = '';
        
        if (data.detections.length === 0) {
            resultsBody.innerHTML = `<tr><td colspan="4" style="text-align: center; color: var(--text-muted);">No compliance objects detected in this scene.</td></tr>`;
            resultsTableContainer.classList.remove('hidden');
            return;
        }
        
        let hasViolations = false;
        
        data.detections.forEach(det => {
            const [rawX, rawY, rawW, rawH] = det.box;
            
            // Scaled coordinates
            const x = rawX * scaleX;
            const y = rawY * scaleY;
            const w = rawW * scaleX;
            const h = rawH * scaleY;
            
            // Design style mapping by class
            let color, badgeClass, severity, severityClass;
            
            if (det.class === 'Helmet') {
                color = '#10b981'; // Emerald
                badgeClass = 'helmet';
                severity = 'Compliant';
                severityClass = 'info';
            } else if (det.class === 'No_Helmet') {
                color = '#ef4444'; // Red
                badgeClass = 'no-helmet';
                severity = 'Violation';
                severityClass = 'critical';
                hasViolations = true;
            } else if (det.class === 'Seatbelt') {
                color = '#10b981'; // Emerald
                badgeClass = 'seatbelt';
                severity = 'Compliant';
                severityClass = 'info';
            } else if (det.class === 'No_Seatbelt') {
                color = '#ef4444'; // Red
                badgeClass = 'no-seatbelt';
                severity = 'Violation';
                severityClass = 'critical';
                hasViolations = true;
            } else { // License_Plate
                color = '#00f0ff'; // Neon Cyan
                badgeClass = 'license-plate';
                severity = 'Identifier';
                severityClass = 'info';
            }
            
            // Draw box outline
            ctx.strokeStyle = color;
            ctx.lineWidth = 3;
            ctx.strokeRect(x, y, w, h);
            
            // Draw glowing inner shadow effect
            ctx.shadowColor = color;
            ctx.shadowBlur = 4;
            ctx.strokeStyle = color;
            ctx.lineWidth = 1;
            ctx.strokeRect(x, y, w, h);
            
            // Reset shadows for text drawing
            ctx.shadowBlur = 0;
            
            // Draw text badge label background
            const label = `${det.class} ${Math.round(det.confidence * 100)}%`;
            ctx.font = 'bold 12px Inter, system-ui';
            const textWidth = ctx.measureText(label).width;
            
            ctx.fillStyle = color;
            ctx.fillRect(x - 1.5, y - 20, textWidth + 12, 20);
            
            // Draw text
            ctx.fillStyle = '#0a0d16'; // Base dark body text
            ctx.fillText(label, x + 4, y - 6);
            
            // Add Row to Results Table
            const row = document.createElement('tr');
            row.innerHTML = `
                <td><span class="badge-class ${badgeClass}">${det.class}</span></td>
                <td><strong>${Math.round(det.confidence * 100)}%</strong></td>
                <td><code>[${det.box.join(', ')}]</code></td>
                <td><span class="badge-severity ${severityClass}">${severity}</span></td>
            `;
            resultsBody.appendChild(row);
        });
        
        // Show table
        resultsTableContainer.classList.remove('hidden');
        
        // Update dashboard violations counter if a violation was found
        if (hasViolations) {
            detectionsLogged += 1;
            totalViolationsCounter.textContent = detectionsLogged.toLocaleString();
            totalViolationsCounter.classList.add('text-red');
        }
    }
    
    // ----------------------------------------------------
    // 5. Live CCTV Simulation (Canvas Animation Loop)
    // ----------------------------------------------------
    const cctvCtx = cctvSimCanvas.getContext('2d');
    let frameCount = 0;
    
    // Object States
    let riderX = -40;
    let hasHelmet = false;
    let carX = 180;
    let hasSeatbelt = true;
    
    function updateCctvStream() {
        if (!isCctvRunning) return;
        
        // Clear canvas with dark futuristic feed color
        cctvCtx.fillStyle = '#0f111a';
        cctvCtx.fillRect(0, 0, 480, 270);
        
        // Draw moving digital grid background (sci-fi surveillance theme)
        cctvCtx.strokeStyle = 'rgba(0, 240, 255, 0.05)';
        cctvCtx.lineWidth = 1;
        for (let x = 0; x < 480; x += 40) {
            cctvCtx.beginPath();
            cctvCtx.moveTo(x, 0);
            cctvCtx.lineTo(x, 270);
            cctvCtx.stroke();
        }
        for (let y = 0; y < 270; y += 30) {
            cctvCtx.beginPath();
            cctvCtx.moveTo(0, y);
            cctvCtx.lineTo(480, y);
            cctvCtx.stroke();
        }

        // Draw roadway lane perspective
        cctvCtx.fillStyle = '#1e2130';
        cctvCtx.beginPath();
        cctvCtx.moveTo(60, 270);
        cctvCtx.lineTo(190, 80);
        cctvCtx.lineTo(250, 80);
        cctvCtx.lineTo(380, 270);
        cctvCtx.closePath();
        cctvCtx.fill();
        
        // Draw green side lanes (compliance boundaries)
        cctvCtx.strokeStyle = 'rgba(16, 185, 129, 0.3)';
        cctvCtx.lineWidth = 2;
        cctvCtx.beginPath();
        cctvCtx.moveTo(60, 270);
        cctvCtx.lineTo(190, 80);
        cctvCtx.moveTo(380, 270);
        cctvCtx.lineTo(250, 80);
        cctvCtx.stroke();

        // Moving center dashes (Perspective Correct)
        const scrollSpeed = 4;
        const dashOffset = (frameCount * scrollSpeed) % 40;
        cctvCtx.strokeStyle = 'rgba(255, 255, 255, 0.7)';
        cctvCtx.lineWidth = 3;
        cctvCtx.beginPath();
        for (let i = 0; i < 8; i++) {
            const currentY = 80 + i * 25 + dashOffset;
            if (currentY > 270) continue;
            const progress = (currentY - 80) / 190; // 0 to 1
            const x = 220 + progress * 5;
            const dashLength = 3 + progress * 18;
            cctvCtx.moveTo(x, currentY);
            cctvCtx.lineTo(x, currentY + dashLength);
        }
        cctvCtx.stroke();

        // Move Car down the lane (coming closer, scaling up)
        carX += 0.85;
        if (carX > 320) {
            carX = 140;
            hasSeatbelt = !hasSeatbelt; // Alternate seatbelt compliance status
        }
        
        const carProgress = (carX - 140) / 180; // 0 to 1
        const carSizeW = 35 + carProgress * 70;
        const carSizeH = 20 + carProgress * 45;
        const carY = 90 + carProgress * 140;
        const carScreenX = 260 + carProgress * 60;
        
        // Draw Car Shadow
        cctvCtx.fillStyle = 'rgba(0, 0, 0, 0.4)';
        cctvCtx.beginPath();
        cctvCtx.ellipse(carScreenX, carY + carSizeH - 2, carSizeW * 0.5, 4, 0, 0, Math.PI * 2);
        cctvCtx.fill();

        // Draw Car Body (sleeker look with gradient)
        const carGrad = cctvCtx.createLinearGradient(carScreenX - carSizeW/2, carY, carScreenX + carSizeW/2, carY + carSizeH);
        carGrad.addColorStop(0, '#3b82f6');
        carGrad.addColorStop(1, '#1d4ed8');
        cctvCtx.fillStyle = carGrad;
        cctvCtx.fillRect(carScreenX - carSizeW/2, carY, carSizeW, carSizeH);
        
        // Draw Car Windows
        cctvCtx.fillStyle = 'rgba(147, 197, 253, 0.85)';
        cctvCtx.fillRect(carScreenX - carSizeW/2 + 6, carY + 3, carSizeW - 12, carSizeH/3.5);

        // Draw Car Wheels (Side profile / perspective circles)
        cctvCtx.fillStyle = '#0f172a';
        cctvCtx.fillRect(carScreenX - carSizeW/2 + 2, carY + carSizeH - 4, carSizeW/5, 4);
        cctvCtx.fillRect(carScreenX + carSizeW/2 - 2 - carSizeW/5, carY + carSizeH - 4, carSizeW/5, 4);

        // Draw Driver Silhouette in Window
        const driverX = carScreenX - carSizeW/6;
        const driverY = carY + carSizeH/4;
        const driverRadius = carSizeH/8;
        
        cctvCtx.fillStyle = '#0f172a';
        cctvCtx.beginPath();
        cctvCtx.arc(driverX, driverY, driverRadius, 0, Math.PI * 2);
        cctvCtx.fill();
        cctvCtx.fillRect(driverX - driverRadius, driverY + driverRadius, driverRadius * 2, driverRadius);
        
        // Draw Seatbelt line / Bounding box
        if (hasSeatbelt) {
            cctvCtx.strokeStyle = '#10b981'; // Green
            cctvCtx.lineWidth = 1.5;
            cctvCtx.beginPath();
            cctvCtx.moveTo(driverX - driverRadius, driverY - 2);
            cctvCtx.lineTo(driverX + driverRadius, driverY + driverRadius * 1.8);
            cctvCtx.stroke();
            
            // Green HUD target brackets
            cctvCtx.strokeStyle = '#10b981';
            cctvCtx.lineWidth = 1.5;
            const bx = driverX - driverRadius - 3;
            const by = driverY - driverRadius - 3;
            const bw = driverRadius * 2 + 6;
            const bh = driverRadius * 2.5 + 6;
            drawHUDBrackets(bx, by, bw, bh, '#10b981');
            
            cctvCtx.fillStyle = '#10b981';
            cctvCtx.font = '6px "Fira Code", monospace';
            cctvCtx.fillText("Seatbelt:OK (92%)", bx, by - 4);
        } else {
            // Red HUD target brackets (Violation)
            cctvCtx.strokeStyle = '#ef4444';
            cctvCtx.lineWidth = 1.5;
            const bx = driverX - driverRadius - 3;
            const by = driverY - driverRadius - 3;
            const bw = driverRadius * 2 + 6;
            const bh = driverRadius * 2.5 + 6;
            drawHUDBrackets(bx, by, bw, bh, '#ef4444');
            
            cctvCtx.fillStyle = '#ef4444';
            cctvCtx.font = '6px "Fira Code", monospace';
            cctvCtx.fillText("No_Seatbelt! (87%)", bx, by - 4);
        }
        
        // Draw License Plate (White rect)
        const plateW = 12 + carProgress * 15;
        const plateH = 5 + carProgress * 5;
        const plateX = carScreenX - plateW/2;
        const plateY = carY + carSizeH - plateH - 2;
        
        cctvCtx.fillStyle = '#f8fafc';
        cctvCtx.fillRect(plateX, plateY, plateW, plateH);
        
        // Bounding box for plate (Cyan HUD brackets)
        drawHUDBrackets(plateX - 2, plateY - 2, plateW + 4, plateH + 4, '#00f0ff');
        cctvCtx.fillStyle = '#00f0ff';
        cctvCtx.font = '6px "Fira Code", monospace';
        cctvCtx.fillText("Plate:SF-96-AI", plateX - 2, plateY - 5);
        
        // Move Motorbike Rider
        riderX += 1.6;
        if (riderX > 520) {
            riderX = -40;
            hasHelmet = !hasHelmet; // Alternate compliance status
        }
        
        const riderProgress = riderX / 520;
        const riderY = 240 - riderProgress * 140;
        const riderScale = 0.5 + (1 - riderProgress) * 0.8; // scaling
        const rW = 20 * riderScale;
        const rH = 40 * riderScale;
        
        // Draw Motorcyclist Shadow
        cctvCtx.fillStyle = 'rgba(0, 0, 0, 0.4)';
        cctvCtx.beginPath();
        cctvCtx.ellipse(riderX, riderY + 2, rW * 0.7, 3 * riderScale, 0, 0, Math.PI * 2);
        cctvCtx.fill();

        // Draw Motorcyclist body
        cctvCtx.fillStyle = '#ec4899';
        cctvCtx.fillRect(riderX - rW/2, riderY - rH, rW, rH);
        
        // Draw Wheels
        cctvCtx.fillStyle = '#090d16';
        cctvCtx.beginPath();
        cctvCtx.arc(riderX - rW/3, riderY, 6*riderScale, 0, Math.PI * 2);
        cctvCtx.arc(riderX + rW/3, riderY, 6*riderScale, 0, Math.PI * 2);
        cctvCtx.fill();
        
        // Head / Helmet area
        const headRadius = 5 * riderScale;
        const headX = riderX;
        const headY = riderY - rH;
        
        if (hasHelmet) {
            // Draw Helmet (Green circle)
            cctvCtx.fillStyle = '#10b981';
            cctvCtx.beginPath();
            cctvCtx.arc(headX, headY, headRadius + 1, 0, Math.PI * 2);
            cctvCtx.fill();
            
            // Green HUD brackets
            const bx = headX - headRadius - 2;
            const by = headY - headRadius - 2;
            const bw = headRadius * 2 + 4;
            const bh = headRadius * 2 + 4;
            drawHUDBrackets(bx, by, bw, bh, '#10b981');
            
            cctvCtx.fillStyle = '#10b981';
            cctvCtx.font = '6px "Fira Code", monospace';
            cctvCtx.fillText("Helmet:OK (91%)", bx, by - 5);
            cctvAlert.classList.add('hidden');
        } else {
            // Draw Face (Skin colored circle)
            cctvCtx.fillStyle = '#fbcfe8';
            cctvCtx.beginPath();
            cctvCtx.arc(headX, headY, headRadius, 0, Math.PI * 2);
            cctvCtx.fill();
            
            // Red HUD brackets (Violation)
            const bx = headX - headRadius - 2;
            const by = headY - headRadius - 2;
            const bw = headRadius * 2 + 4;
            const bh = headRadius * 2 + 4;
            drawHUDBrackets(bx, by, bw, bh, '#ef4444');
            
            cctvCtx.fillStyle = '#ef4444';
            cctvCtx.font = '6px "Fira Code", monospace';
            cctvCtx.fillText("No_Helmet! (88%)", bx, by - 5);
            
            // Only trigger alert when rider is in center frame
            if (riderX > 150 && riderX < 350) {
                cctvAlert.classList.remove('hidden');
                cctvAlert.innerHTML = `<span>ALERT: NO_HELMET VIOLATION DETECTED</span>`;
            } else {
                cctvAlert.classList.add('hidden');
            }
        }

        // Trigger seatbelt alert if driver has no seatbelt and car is close
        const isCarSeatbeltAlert = (!hasSeatbelt && carX > 200 && carX < 280);
        if (isCarSeatbeltAlert) {
            cctvAlert.classList.remove('hidden');
            cctvAlert.innerHTML = `<span>ALERT: NO_SEATBELT VIOLATION DETECTED</span>`;
        }

        // Pulse warning border on the entire canvas if any alert is active
        const isAlertActive = (!cctvAlert.classList.contains('hidden'));
        if (isAlertActive) {
            cctvCtx.strokeStyle = `rgba(239, 68, 68, ${0.3 + Math.sin(frameCount * 0.15) * 0.2})`;
            cctvCtx.lineWidth = 4;
            cctvCtx.strokeRect(0, 0, 480, 270);
        }

        // Draw HUD Camera Overlays (crosshair, guidelines)
        cctvCtx.strokeStyle = 'rgba(0, 240, 255, 0.2)';
        cctvCtx.lineWidth = 1;
        // Central Crosshair
        cctvCtx.beginPath();
        cctvCtx.moveTo(240, 125); cctvCtx.lineTo(240, 145);
        cctvCtx.moveTo(230, 135); cctvCtx.lineTo(250, 135);
        cctvCtx.stroke();
        
        // Outer corners brackets
        drawHUDBrackets(10, 10, 460, 250, 'rgba(0, 240, 255, 0.4)', 15);

        // Fluctuate Seatbelt Compliance Rate slightly for live effect
        if (frameCount % 120 === 0) {
            const fluctuation = (93.5 + Math.random() * 2).toFixed(1);
            const valEl = document.getElementById('seatbeltComplianceVal');
            if (valEl) {
                valEl.textContent = fluctuation + '%';
                valEl.nextElementSibling.firstElementChild.style.width = fluctuation + '%';
            }
        }
        
        // Update Time display dynamically
        const now = new Date();
        const timeStr = now.toLocaleDateString('en-US') + ' ' + now.toLocaleTimeString('en-US');
        cctvTime.textContent = timeStr;
        
        frameCount++;
        requestAnimationFrame(updateCctvStream);
    }

    // Helper to draw clean Sci-Fi HUD brackets instead of simple boxes
    function drawHUDBrackets(x, y, w, h, color, size = 4) {
        cctvCtx.strokeStyle = color;
        cctvCtx.lineWidth = 1.5;
        // Top Left
        cctvCtx.beginPath();
        cctvCtx.moveTo(x + size, y); cctvCtx.lineTo(x, y); cctvCtx.lineTo(x, y + size);
        // Top Right
        cctvCtx.moveTo(x + w - size, y); cctvCtx.lineTo(x + w, y); cctvCtx.lineTo(x + w, y + size);
        // Bottom Left
        cctvCtx.moveTo(x + size, y + h); cctvCtx.lineTo(x, y + h); cctvCtx.lineTo(x, y + h - size);
        // Bottom Right
        cctvCtx.moveTo(x + w - size, y + h); cctvCtx.lineTo(x + w, y + h); cctvCtx.lineTo(x + w, y + h - size);
        cctvCtx.stroke();
    }
    
    // Start simulation loop
    requestAnimationFrame(updateCctvStream);
    
    // CCTV Controls Play / Pause
    cctvPlayBtn.addEventListener('click', () => {
        isCctvRunning = !isCctvRunning;
        cctvPlayBtn.textContent = isCctvRunning ? 'Pause Feed' : 'Resume Feed';
        if (isCctvRunning) {
            requestAnimationFrame(updateCctvStream);
        }
    });

    // ----------------------------------------------------
    // 6. Quick Test Gallery Initialization
    // ----------------------------------------------------
    const galleryGrid = document.getElementById('galleryGrid');
    const gallerySection = document.getElementById('gallerySection');
    
    function initializeGallery() {
        if (!galleryGrid) return;
        for (let i = 1; i <= 6; i++) {
            const numStr = String(i).padStart(2, '0');
            const filename = `sample_${numStr}.png`;
            const url = `/static/samples/${filename}`;
            
            const item = document.createElement('div');
            item.className = 'gallery-item';
            item.innerHTML = `
                <img src="${url}" alt="Scenario ${numStr}">
                <div class="gallery-item-label">Scenario ${numStr}</div>
            `;
            item.addEventListener('click', () => {
                selectSampleImage(url, filename);
            });
            galleryGrid.appendChild(item);
        }
    }
    
    function selectSampleImage(url, filename) {
        // Show loading state in inspector
        jsonResponse.textContent = `Loading ${filename}...\nPreparing test payload...`;
        
        fetch(url)
            .then(res => {
                if (!res.ok) throw new Error("Sample file not found");
                return res.blob();
            })
            .then(blob => {
                const file = new File([blob], filename, { type: "image/jpeg" });
                handleUploadedFile(file);
            })
            .catch(err => {
                console.error("Error loading sample image:", err);
                alert("Failed to load sample image. Please ensure the server has run 'generate_samples.py' first.");
            });
    }
    
    initializeGallery();
});
