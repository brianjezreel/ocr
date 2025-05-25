document.addEventListener('DOMContentLoaded', () => {
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const startButton = document.getElementById('startCamera');
    const captureButton = document.getElementById('captureImage');
    const fileInput = document.getElementById('fileInput');
    const result = document.getElementById('result');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const previewImage = document.getElementById('previewImage');
    const boxesCanvas = document.getElementById('boxesCanvas');
    const boxesCtx = boxesCanvas.getContext('2d');
    const showBoxesCheckbox = document.getElementById('showBoxes');
    const confidenceThreshold = document.getElementById('confidenceThreshold');
    const confidenceValue = document.getElementById('confidenceValue');
    const groupThreshold = document.getElementById('groupThreshold');
    const groupValue = document.getElementById('groupValue');
    const textTooltip = document.getElementById('textTooltip');
    const preprocessingView = document.getElementById('preprocessingView');
    
    let currentBoxes = [];
    let stream = null;
    let currentPreprocessedImages = {};

    // Initialize box visibility
    boxesCanvas.style.display = showBoxesCheckbox.checked ? 'block' : 'none';

    // Update confidence threshold display
    confidenceThreshold.addEventListener('input', (e) => {
        confidenceValue.textContent = `${e.target.value}%`;
        if (currentBoxes.length > 0 && showBoxesCheckbox.checked) {
            drawBoxes(currentBoxes);
        }
    });

    // Update group threshold display
    groupThreshold.addEventListener('input', (e) => {
        groupValue.textContent = `${e.target.value}px`;
        if (currentBoxes.length > 0 && showBoxesCheckbox.checked) {
            drawBoxes(currentBoxes);
        }
    });

    // Toggle boxes visibility with transition
    showBoxesCheckbox.addEventListener('change', () => {
        if (currentBoxes.length > 0) {
            if (showBoxesCheckbox.checked) {
                boxesCanvas.style.display = 'block';
                // Small delay to allow display change before opacity transition
                requestAnimationFrame(() => {
                    boxesCanvas.style.opacity = '1';
                    drawBoxes(currentBoxes);
                });
            } else {
                boxesCanvas.style.opacity = '0';
                textTooltip.style.display = 'none';
                // Wait for transition before hiding
                setTimeout(() => {
                    if (!showBoxesCheckbox.checked) {
                        boxesCanvas.style.display = 'none';
                    }
                }, 200);
            }
        }
    });

    // Add preprocessing view change handler
    preprocessingView.addEventListener('change', (e) => {
        const selectedView = e.target.value;
        if (currentPreprocessedImages[selectedView]) {
            const previewContainer = document.querySelector('.preview-container');
            previewContainer.classList.add('loading');
            
            // Create a new image to preload
            const tempImage = new Image();
            tempImage.onload = () => {
                previewImage.src = currentPreprocessedImages[selectedView];
                previewImage.style.display = 'block';
                previewContainer.classList.remove('loading');
                
                // Redraw boxes if they exist
                if (currentBoxes.length > 0) {
                    updateCanvasSize();
                    drawBoxes(currentBoxes);
                }
            };
            tempImage.src = currentPreprocessedImages[selectedView];
        }
    });

    function updateCanvasSize() {
        // Get the natural dimensions of the image
        const naturalWidth = previewImage.naturalWidth;
        const naturalHeight = previewImage.naturalHeight;
        
        // Get the displayed dimensions of the image
        const displayedWidth = previewImage.offsetWidth;
        const displayedHeight = previewImage.offsetHeight;

        // Set canvas dimensions to match displayed image size
        boxesCanvas.style.width = `${displayedWidth}px`;
        boxesCanvas.style.height = `${displayedHeight}px`;
        
        // Set actual canvas dimensions to match displayed size
        boxesCanvas.width = displayedWidth;
        boxesCanvas.height = displayedHeight;

        // Position canvas exactly over the image
        const imageRect = previewImage.getBoundingClientRect();
        boxesCanvas.style.top = '0px';
        boxesCanvas.style.left = '0px';

        // Store scale factors for coordinate conversion
        boxesCanvas.scaleX = displayedWidth / naturalWidth;
        boxesCanvas.scaleY = displayedHeight / naturalHeight;
        
        // Redraw boxes if they exist
        if (currentBoxes.length > 0) {
            drawBoxes(currentBoxes);
        }
    }

    // Add resize observer to handle image size changes
    const resizeObserver = new ResizeObserver((entries) => {
        for (const entry of entries) {
            if (entry.target === previewImage && previewImage.style.display !== 'none') {
                updateCanvasSize();
            }
        }
    });
    resizeObserver.observe(previewImage);

    function displayPreview(source) {
        previewImage.src = source;
        previewImage.style.display = 'block';
        
        // Reset preprocessing view to original
        preprocessingView.value = 'original';
        
        // Clear previous boxes and preprocessed images
        currentBoxes = [];
        currentPreprocessedImages = {};
        
        // Wait for image to load before setting canvas size
        previewImage.onload = () => {
            updateCanvasSize();
        };
    }

    function groupBoxes(boxes) {
        const threshold = parseInt(groupThreshold.value);
        if (threshold === 0) return boxes;

        // Sort boxes by line and position
        boxes.sort((a, b) => {
            if (a.block_num !== b.block_num) return a.block_num - b.block_num;
            if (a.line_num !== b.line_num) return a.line_num - b.line_num;
            return a.x - b.x;
        });

        const grouped = [];
        let currentGroup = [];
        let lastBox = null;

        boxes.forEach(box => {
            if (!lastBox) {
                currentGroup = [box];
            } else {
                const xDist = Math.abs((lastBox.x + lastBox.width) - box.x) * boxesCanvas.width;
                const sameLine = lastBox.line_num === box.line_num && lastBox.block_num === box.block_num;
                
                if (sameLine && xDist < threshold) {
                    currentGroup.push(box);
                } else {
                    if (currentGroup.length > 0) {
                        grouped.push(mergeGroup(currentGroup));
                    }
                    currentGroup = [box];
                }
            }
            lastBox = box;
        });

        if (currentGroup.length > 0) {
            grouped.push(mergeGroup(currentGroup));
        }

        return grouped;
    }

    function mergeGroup(group) {
        return {
            x: Math.min(...group.map(b => b.x)),
            y: Math.min(...group.map(b => b.y)),
            width: Math.max(...group.map(b => b.x + b.width)) - Math.min(...group.map(b => b.x)),
            height: Math.max(...group.map(b => b.y + b.height)) - Math.min(...group.map(b => b.y)),
            text: group.map(b => b.text).join(' '),
            conf: Math.min(...group.map(b => b.conf)),
            block_num: group[0].block_num,
            line_num: group[0].line_num
        };
    }

    function drawBoxes(boxes) {
        // Clear canvas first
        boxesCtx.clearRect(0, 0, boxesCanvas.width, boxesCanvas.height);

        // If boxes are hidden, just clear the canvas and return
        if (!showBoxesCheckbox.checked) {
            boxesCanvas.style.display = 'none';
            textTooltip.style.display = 'none';
            return;
        }

        // Show canvas if it was hidden
        boxesCanvas.style.display = 'block';

        const minConfidence = parseInt(confidenceThreshold.value);
        
        // Filter and group boxes
        const filteredBoxes = boxes.filter(box => box.conf >= minConfidence);
        const groupedBoxes = groupBoxes(filteredBoxes);

        // Set drawing styles
        boxesCtx.strokeStyle = '#2ecc71';
        boxesCtx.lineWidth = 2;
        boxesCtx.fillStyle = 'rgba(46, 204, 113, 0.1)';

        groupedBoxes.forEach(box => {
            // Convert relative coordinates to actual canvas coordinates
            const x = Math.round(box.x * boxesCanvas.width);
            const y = Math.round(box.y * boxesCanvas.height);
            const width = Math.round(box.width * boxesCanvas.width);
            const height = Math.round(box.height * boxesCanvas.height);

            // Draw filled rectangle with transparency
            boxesCtx.fillRect(x, y, width, height);
            
            // Draw border
            boxesCtx.strokeRect(x, y, width, height);
        });

        // Remove any existing listeners before adding new ones
        boxesCanvas.removeEventListener('mousemove', handleMouseMove);
        boxesCanvas.removeEventListener('mouseleave', handleMouseLeave);

        // Handle mouse interactions
        function handleMouseMove(e) {
            if (!showBoxesCheckbox.checked) return;

            const rect = boxesCanvas.getBoundingClientRect();
            const scaleX = boxesCanvas.width / rect.width;
            const scaleY = boxesCanvas.height / rect.height;
            
            const x = (e.clientX - rect.left) * scaleX / boxesCanvas.width;
            const y = (e.clientY - rect.top) * scaleY / boxesCanvas.height;

            // Find box under cursor
            const box = groupedBoxes.find(b => 
                x >= b.x && x <= b.x + b.width &&
                y >= b.y && y <= b.y + b.height
            );

            if (box) {
                textTooltip.style.display = 'block';
                textTooltip.style.left = `${e.pageX + 10}px`;
                textTooltip.style.top = `${e.pageY + 10}px`;
                textTooltip.textContent = `${box.text} (${Math.round(box.conf)}%)`;
            } else {
                textTooltip.style.display = 'none';
            }
        }

        function handleMouseLeave() {
            textTooltip.style.display = 'none';
        }

        boxesCanvas.addEventListener('mousemove', handleMouseMove);
        boxesCanvas.addEventListener('mouseleave', handleMouseLeave);
    }

    // Camera handling
    startButton.addEventListener('click', async () => {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ 
                video: { 
                    facingMode: 'environment',
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                } 
            });
            video.srcObject = stream;
            startButton.style.display = 'none';
            captureButton.disabled = false;
        } catch (err) {
            console.error('Error accessing camera:', err);
            result.textContent = 'Error accessing camera. Please ensure camera permissions are granted.';
        }
    });

    captureButton.addEventListener('click', () => {
        if (stream) {
            // Set canvas dimensions to match video
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            
            // Draw video frame to canvas
            const context = canvas.getContext('2d');
            context.drawImage(video, 0, 0, canvas.width, canvas.height);
            
            // Display preview
            displayPreview(canvas.toDataURL('image/jpeg'));
            
            // Convert canvas to blob and process
            canvas.toBlob((blob) => {
                // Create a File object from the blob
                const file = new File([blob], 'camera-capture.jpg', { type: 'image/jpeg' });
                processImage(file);
            }, 'image/jpeg', 0.8);
            
            // Stop camera stream
            stream.getTracks().forEach(track => track.stop());
            video.srcObject = null;
            stream = null;
            
            // Reset buttons
            startButton.style.display = 'block';
            captureButton.disabled = true;
        }
    });

    // File upload handling
    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            // Display preview
            const reader = new FileReader();
            reader.onload = (e) => displayPreview(e.target.result);
            reader.readAsDataURL(file);
            
            processImage(file);
        }
    });

    // Drag and drop handling
    const dropZone = document.querySelector('.file-upload');
    
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = '#3498db';
    });

    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = '#bdc3c7';
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = '#bdc3c7';
        
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            // Display preview
            const reader = new FileReader();
            reader.onload = (e) => displayPreview(e.target.result);
            reader.readAsDataURL(file);
            
            processImage(file);
        } else {
            result.textContent = 'Please drop an image file.';
        }
    });

    // Process image and send to server
    async function processImage(blob) {
        const formData = new FormData();
        formData.append('image', blob);

        try {
            loadingSpinner.style.display = 'block';
            result.textContent = '';

            const response = await fetch('/process-image', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.error) {
                result.textContent = `Error: ${data.error}`;
            } else {
                // Store preprocessed images
                currentPreprocessedImages = data.preprocessed_images;
                
                // Display text in a <pre> tag to preserve all spaces and line breaks
                result.innerHTML = `<pre>${data.text}</pre>`;
                
                if (data.boxes && data.boxes.length > 0) {
                    currentBoxes = data.boxes;
                    drawBoxes(data.boxes);
                }
            }
        } catch (err) {
            result.textContent = 'Error processing image. Please try again.';
            console.error('Error:', err);
        } finally {
            loadingSpinner.style.display = 'none';
        }
    }
}); 