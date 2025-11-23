/**
 * Face Detection Module using TensorFlow.js Face Landmarks Detection
 */

class FaceDetector {
    constructor() {
        this.model = null;
        this.video = null;
        this.canvas = null;
        this.ctx = null;
        this.isRunning = false;

        // Calibration
        this.calibrated = false;
        this.calibrationFrames = 0;
        this.calibrationNeeded = 30;
        this.baselineBAR = 0;
        this.sumBAR = 0;
        this.baselineFaceX = 0;
        this.sumFaceX = 0;

        // Thresholds
        this.eyebrowRaiseThreshold = 0;

        // State tracking
        this.eyebrowsWereRaised = false;
        this.lastEyebrowTime = 0;
        this.eyebrowCooldown = 400; // ms

        // Callbacks
        this.onCalibrationProgress = null;
        this.onCalibrationComplete = null;
        this.onFaceDetected = null;
        this.onNoFace = null;
    }

    async init(videoElement, canvasElement) {
        this.video = videoElement;
        this.canvas = canvasElement;
        this.ctx = canvasElement.getContext('2d');

        console.log('Init called with:', { video: !!videoElement, canvas: !!canvasElement });

        // Load TensorFlow.js and Face Landmarks Detection
        console.log('Loading TensorFlow.js models...');

        try {
            // Load the face landmarks detection model
            this.model = await faceLandmarksDetection.createDetector(
                faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh,
                {
                    runtime: 'tfjs',
                    refineLandmarks: true,
                    maxFaces: 1
                }
            );
            console.log('Face detection model loaded!');
            return true;
        } catch (error) {
            console.error('Error loading model:', error);
            return false;
        }
    }

    async startCamera() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'user'
                }
            });
            this.video.srcObject = stream;

            // Wait for video metadata to load
            await new Promise((resolve) => {
                this.video.onloadedmetadata = () => {
                    resolve();
                };
            });

            await this.video.play();

            // Set canvas internal size to match video natural dimensions
            // CSS handles the display scaling via position: absolute with top/left/right/bottom: 0
            this.canvas.width = this.video.videoWidth;
            this.canvas.height = this.video.videoHeight;

            console.log('Camera started successfully!');
            console.log('Video dimensions:', this.video.videoWidth, 'x', this.video.videoHeight);
            console.log('Canvas dimensions:', this.canvas.width, 'x', this.canvas.height);
            console.log('Video playing:', !this.video.paused);

            return true;
        } catch (error) {
            console.error('Error accessing camera:', error);
            return false;
        }
    }

    calculateDistance(p1, p2) {
        return Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2));
    }

    // Get eyebrow aspect ratio using MediaPipe Face Mesh landmarks
    getEyebrowAspectRatio(keypoints) {
        // MediaPipe Face Mesh has 478 landmarks (with refineLandmarks: true)
        // Right eyebrow top: 66, 105, 63, 70
        // Left eyebrow top: 296, 334, 293, 300
        // Right eye upper lid: 159, 158, 157
        // Left eye upper lid: 386, 385, 384

        // Use multiple points for more accurate measurement
        // Right side: eyebrow center (105) to eye upper (159)
        const rightEyebrow1 = keypoints[66];
        const rightEyebrow2 = keypoints[105];
        const rightEyebrow3 = keypoints[63];
        const rightEyeUpper = keypoints[159];

        // Left side: eyebrow center (334) to eye upper (386)
        const leftEyebrow1 = keypoints[296];
        const leftEyebrow2 = keypoints[334];
        const leftEyebrow3 = keypoints[293];
        const leftEyeUpper = keypoints[386];

        // Calculate average eyebrow Y position
        const rightEyebrowY = (rightEyebrow1.y + rightEyebrow2.y + rightEyebrow3.y) / 3;
        const leftEyebrowY = (leftEyebrow1.y + leftEyebrow2.y + leftEyebrow3.y) / 3;

        // Calculate heights (eyebrow to eye distance)
        const rightHeight = rightEyeUpper.y - rightEyebrowY;
        const leftHeight = leftEyeUpper.y - leftEyebrowY;

        // Normalize by face height (forehead to chin)
        const forehead = keypoints[10];
        const chin = keypoints[152];
        const faceHeight = this.calculateDistance(forehead, chin);

        const rightBAR = rightHeight / faceHeight;
        const leftBAR = leftHeight / faceHeight;

        return (rightBAR + leftBAR) / 2;
    }

    // Get face center X position (normalized 0-1)
    getFaceCenterX(keypoints) {
        // Use nose tip as face center reference
        // CSS mirrors the video, so when you move LEFT, the keypoint.x increases
        // We return normalized position where 0 = left side, 1 = right side
        const noseTip = keypoints[1];
        return noseTip.x / this.video.videoWidth;
    }

    // Get Eye Aspect Ratio for blink/expression detection
    getEyeAspectRatio(keypoints) {
        // Right eye landmarks
        const rightEyeTop = keypoints[159];
        const rightEyeBottom = keypoints[145];
        const rightEyeLeft = keypoints[33];
        const rightEyeRight = keypoints[133];

        const rightHeight = this.calculateDistance(rightEyeTop, rightEyeBottom);
        const rightWidth = this.calculateDistance(rightEyeLeft, rightEyeRight);
        const rightEAR = rightHeight / rightWidth;

        // Left eye landmarks
        const leftEyeTop = keypoints[386];
        const leftEyeBottom = keypoints[374];
        const leftEyeLeft = keypoints[362];
        const leftEyeRight = keypoints[263];

        const leftHeight = this.calculateDistance(leftEyeTop, leftEyeBottom);
        const leftWidth = this.calculateDistance(leftEyeLeft, leftEyeRight);
        const leftEAR = leftHeight / leftWidth;

        return {
            left: leftEAR,
            right: rightEAR,
            average: (leftEAR + rightEAR) / 2
        };
    }

    // Get Mouth Aspect Ratio
    getMouthAspectRatio(keypoints) {
        const upperLip = keypoints[13];
        const lowerLip = keypoints[14];
        const leftCorner = keypoints[61];
        const rightCorner = keypoints[291];

        const height = this.calculateDistance(upperLip, lowerLip);
        const width = this.calculateDistance(leftCorner, rightCorner);

        return height / width;
    }

    async detectFace() {
        if (!this.model || !this.video) {
            console.log('Model or video not ready:', { model: !!this.model, video: !!this.video });
            return null;
        }

        try {
            const predictions = await this.model.estimateFaces(this.video);
            console.log('Predictions:', predictions.length);

            if (predictions.length > 0) {
                return predictions[0];
            }
            return null;
        } catch (error) {
            console.error('Detection error:', error);
            return null;
        }
    }

    drawLandmarks(keypoints) {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // Debug: log canvas dimensions and first keypoint
        console.log('Canvas size:', this.canvas.width, 'x', this.canvas.height);
        console.log('First keypoint:', keypoints[0]?.x, keypoints[0]?.y);

        // Test: draw a visible rectangle to confirm canvas is working
        this.ctx.fillStyle = 'rgba(255, 0, 0, 0.5)';
        this.ctx.fillRect(10, 10, 100, 50);

        // Both video and canvas are mirrored via CSS transform: scaleX(-1)
        // So we draw directly using the keypoint coordinates

        // Draw movement zones
        const leftZoneX = this.canvas.width * 0.42;   // Left zone line
        const rightZoneX = this.canvas.width * 0.58;  // Right zone line

        this.ctx.strokeStyle = 'rgba(255, 255, 0, 0.5)';
        this.ctx.lineWidth = 2;
        this.ctx.setLineDash([5, 5]);

        // Zone lines
        this.ctx.beginPath();
        this.ctx.moveTo(leftZoneX, 0);
        this.ctx.lineTo(leftZoneX, this.canvas.height);
        this.ctx.stroke();

        this.ctx.beginPath();
        this.ctx.moveTo(rightZoneX, 0);
        this.ctx.lineTo(rightZoneX, this.canvas.height);
        this.ctx.stroke();

        this.ctx.setLineDash([]);

        // Zone labels (these will appear mirrored, so swap positions)
        this.ctx.fillStyle = 'rgba(255, 255, 0, 0.8)';
        this.ctx.font = '14px Arial';
        this.ctx.fillText('RIGHT', 10, 30);
        this.ctx.fillText('LEFT', this.canvas.width - 50, 30);

        // Draw eyebrow points
        const eyebrowIndices = [66, 105, 63, 70, 46, 53, 52, 65, 55, 296, 334, 293, 300, 276, 283, 282, 295, 285];
        this.ctx.fillStyle = '#00d4ff';
        eyebrowIndices.forEach(i => {
            if (keypoints[i]) {
                const point = keypoints[i];
                this.ctx.beginPath();
                this.ctx.arc(point.x, point.y, 4, 0, 2 * Math.PI);
                this.ctx.fill();
            }
        });

        // Draw eye points
        const eyeIndices = [33, 133, 159, 145, 158, 157, 362, 263, 386, 374, 385, 384];
        this.ctx.fillStyle = '#7b2cbf';
        eyeIndices.forEach(i => {
            if (keypoints[i]) {
                const point = keypoints[i];
                this.ctx.beginPath();
                this.ctx.arc(point.x, point.y, 3, 0, 2 * Math.PI);
                this.ctx.fill();
            }
        });

        // Draw nose (face center indicator)
        const noseTip = keypoints[1];
        if (noseTip) {
            this.ctx.fillStyle = '#00ff00';
            this.ctx.beginPath();
            this.ctx.arc(noseTip.x, noseTip.y, 8, 0, 2 * Math.PI);
            this.ctx.fill();
        }

        // Draw forehead and chin
        const forehead = keypoints[10];
        const chin = keypoints[152];
        if (forehead && chin) {
            this.ctx.fillStyle = '#ff6b6b';
            this.ctx.beginPath();
            this.ctx.arc(forehead.x, forehead.y, 5, 0, 2 * Math.PI);
            this.ctx.fill();
            this.ctx.beginPath();
            this.ctx.arc(chin.x, chin.y, 5, 0, 2 * Math.PI);
            this.ctx.fill();
        }
    }

    calibrate(bar, faceX) {
        if (this.calibrationFrames < this.calibrationNeeded) {
            this.calibrationFrames++;
            this.sumBAR += bar;
            this.sumFaceX += faceX;

            if (this.onCalibrationProgress) {
                this.onCalibrationProgress(this.calibrationFrames, this.calibrationNeeded);
            }
        } else {
            this.baselineBAR = this.sumBAR / this.calibrationNeeded;
            this.baselineFaceX = this.sumFaceX / this.calibrationNeeded;
            this.eyebrowRaiseThreshold = this.baselineBAR * 1.08; // 8% above baseline (more sensitive)
            this.calibrated = true;

            console.log('Calibration complete!');
            console.log('Baseline BAR:', this.baselineBAR.toFixed(4));
            console.log('Threshold:', this.eyebrowRaiseThreshold.toFixed(4));

            if (this.onCalibrationComplete) {
                this.onCalibrationComplete();
            }
        }
    }

    async processFrame() {
        const face = await this.detectFace();

        if (face && face.keypoints) {
            const keypoints = face.keypoints;

            // Calculate metrics
            const bar = this.getEyebrowAspectRatio(keypoints);
            const faceX = this.getFaceCenterX(keypoints);
            const ear = this.getEyeAspectRatio(keypoints);
            const mar = this.getMouthAspectRatio(keypoints);

            // Draw landmarks
            this.drawLandmarks(keypoints);

            if (!this.calibrated) {
                this.calibrate(bar, faceX);
                return {
                    detected: true,
                    calibrating: true,
                    progress: this.calibrationFrames / this.calibrationNeeded
                };
            }

            // Check for eyebrow raise
            const now = Date.now();
            const eyebrowsRaised = bar > this.eyebrowRaiseThreshold;
            let eyebrowAction = false;

            if (eyebrowsRaised && !this.eyebrowsWereRaised) {
                if (now - this.lastEyebrowTime > this.eyebrowCooldown) {
                    eyebrowAction = true;
                    this.lastEyebrowTime = now;
                }
            }
            this.eyebrowsWereRaised = eyebrowsRaised;

            // Determine left/right movement
            // Both video and canvas are CSS mirrored, but keypoints are in original coordinates
            // When user physically moves LEFT: face moves RIGHT in original frame (x increases)
            // When user physically moves RIGHT: face moves LEFT in original frame (x decreases)
            const leftThreshold = 0.58;  // faceX > this = user moved LEFT
            const rightThreshold = 0.42; // faceX < this = user moved RIGHT
            let movement = 'center';

            if (faceX > leftThreshold) {
                movement = 'left';
            } else if (faceX < rightThreshold) {
                movement = 'right';
            }

            // Debug: log face position
            // console.log('faceX:', faceX.toFixed(2), 'movement:', movement);

            const result = {
                detected: true,
                calibrating: false,
                bar: bar,
                baselineBAR: this.baselineBAR,
                threshold: this.eyebrowRaiseThreshold,
                eyebrowsRaised: eyebrowsRaised,
                eyebrowAction: eyebrowAction,
                faceX: faceX,
                movement: movement,
                ear: ear,
                mar: mar,
                keypoints: keypoints
            };

            if (this.onFaceDetected) {
                this.onFaceDetected(result);
            }

            return result;
        } else {
            this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

            if (this.onNoFace) {
                this.onNoFace();
            }

            return {
                detected: false,
                calibrating: false
            };
        }
    }

    resetCalibration() {
        this.calibrated = false;
        this.calibrationFrames = 0;
        this.sumBAR = 0;
        this.sumFaceX = 0;
    }

    stop() {
        this.isRunning = false;
        if (this.video && this.video.srcObject) {
            this.video.srcObject.getTracks().forEach(track => track.stop());
        }
    }
}

// Export for use in other modules
window.FaceDetector = FaceDetector;
