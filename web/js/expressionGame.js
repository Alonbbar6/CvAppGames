/**
 * Expression Matching Game
 */

const EXPRESSIONS = {
    NEUTRAL: { emoji: '&#x1F610;', name: 'NEUTRAL' },
    HAPPY: { emoji: '&#x1F604;', name: 'HAPPY' },
    SURPRISED: { emoji: '&#x1F632;', name: 'SURPRISED' },
    WINK: { emoji: '&#x1F609;', name: 'WINK' }
};

class ExpressionGame {
    constructor() {
        this.score = 0;
        this.round = 1;
        this.totalRounds = 5;
        this.roundDuration = 8000; // 8 seconds
        this.holdDuration = 3000; // 3 seconds to hold
        this.gameState = 'calibrating'; // calibrating, ready, playing, success, failed, gameover

        this.targetExpression = null;
        this.currentExpression = null;
        this.roundStartTime = 0;
        this.holdStartTime = null;
        this.isHolding = false;

        // Baseline values for expression detection
        this.baselineEAR = 0;
        this.baselineBAR = 0;
        this.baselineMAR = 0;

        // Elements
        this.targetEmoji = document.getElementById('targetEmoji');
        this.targetName = document.getElementById('targetName');
        this.timer = document.getElementById('timer');
        this.matchProgress = document.getElementById('matchProgress');
        this.holdTimer = document.getElementById('holdTimer');
        this.roundDisplay = document.getElementById('round');
        this.scoreDisplay = document.getElementById('score');
        this.expressionOverlay = document.getElementById('expressionOverlay');
    }

    startGame() {
        this.gameState = 'ready';
        this.round = 1;
        this.score = 0;
        this.updateScoreDisplay();
        this.nextRound();
    }

    nextRound() {
        if (this.round > this.totalRounds) {
            this.endGame();
            return;
        }

        // Pick random expression
        const expressions = Object.keys(EXPRESSIONS);
        this.targetExpression = expressions[Math.floor(Math.random() * expressions.length)];

        // Update UI
        this.targetEmoji.innerHTML = EXPRESSIONS[this.targetExpression].emoji;
        this.targetName.textContent = EXPRESSIONS[this.targetExpression].name;
        this.roundDisplay.textContent = this.round;

        // Reset state
        this.holdStartTime = null;
        this.isHolding = false;
        this.holdTimer.style.display = 'none';
        this.matchProgress.style.width = '0%';

        // Start round
        this.gameState = 'playing';
        this.roundStartTime = Date.now();
    }

    classifyExpression(data) {
        if (!data.ear || !data.bar || !data.mar) return 'NEUTRAL';

        const earAvg = data.ear.average;
        const earLeft = data.ear.left;
        const earRight = data.ear.right;
        const bar = data.bar;
        const mar = data.mar;

        // Calculate ratios from baseline
        const earRatio = this.baselineEAR > 0 ? earAvg / this.baselineEAR : 1;
        const barRatio = this.baselineBAR > 0 ? bar / this.baselineBAR : 1;
        const marRatio = this.baselineMAR > 0 ? mar / this.baselineMAR : 1;

        // WINK: Asymmetric eyes
        const earDiff = Math.abs(earLeft - earRight);
        if (earDiff > 0.1 && Math.min(earLeft, earRight) < this.baselineEAR * 0.6) {
            return 'WINK';
        }

        // SURPRISED: Wide eyes, raised eyebrows, open mouth
        let surprisedScore = 0;
        if (earRatio > 1.15) surprisedScore++;
        if (barRatio > 1.12) surprisedScore++;
        if (marRatio > 1.3) surprisedScore++;

        if (surprisedScore >= 2) {
            return 'SURPRISED';
        }

        // HAPPY: Smile (mouth wider, slightly open)
        if (marRatio > 1.1 && marRatio < 1.4) {
            return 'HAPPY';
        }

        return 'NEUTRAL';
    }

    calculateMatchScore(currentExpression, targetExpression) {
        if (currentExpression === targetExpression) {
            return 100;
        }
        // Partial credit
        if (currentExpression === 'HAPPY' && targetExpression === 'SURPRISED') {
            return 40;
        }
        if (currentExpression === 'SURPRISED' && targetExpression === 'HAPPY') {
            return 40;
        }
        return 0;
    }

    update(faceData) {
        if (this.gameState === 'calibrating') {
            if (faceData.calibrating) {
                this.expressionOverlay.textContent = 'Calibrating... Keep neutral face';
            } else {
                // Store baseline values
                this.baselineEAR = faceData.ear?.average || 0.3;
                this.baselineBAR = faceData.bar || 0.3;
                this.baselineMAR = faceData.mar || 0.2;
                this.startGame();
            }
            return;
        }

        if (this.gameState !== 'playing') return;

        // Update timer
        const elapsed = Date.now() - this.roundStartTime;
        const remaining = Math.max(0, (this.roundDuration - elapsed) / 1000);
        this.timer.textContent = remaining.toFixed(1) + 's';

        // Time's up
        if (remaining <= 0) {
            this.roundFailed();
            return;
        }

        // Classify expression
        this.currentExpression = this.classifyExpression(faceData);
        const matchScore = this.calculateMatchScore(this.currentExpression, this.targetExpression);

        // Update match bar
        this.matchProgress.style.width = matchScore + '%';
        this.expressionOverlay.textContent = `Your expression: ${this.currentExpression}`;

        // Check for hold
        if (matchScore >= 80) {
            if (!this.isHolding) {
                this.isHolding = true;
                this.holdStartTime = Date.now();
            }

            const holdElapsed = Date.now() - this.holdStartTime;
            const holdRemaining = Math.max(0, (this.holdDuration - holdElapsed) / 1000);

            this.holdTimer.style.display = 'block';
            this.holdTimer.textContent = `HOLD: ${holdRemaining.toFixed(1)}s`;

            if (holdElapsed >= this.holdDuration) {
                this.roundSuccess();
            }
        } else {
            this.isHolding = false;
            this.holdStartTime = null;
            this.holdTimer.style.display = 'none';
        }
    }

    roundSuccess() {
        this.gameState = 'success';
        this.score += 100;
        this.updateScoreDisplay();
        this.expressionOverlay.innerHTML = '&#x2705; SUCCESS! +100 points';
        this.expressionOverlay.style.color = '#00f000';

        setTimeout(() => {
            this.round++;
            this.expressionOverlay.style.color = '#fff';
            this.nextRound();
        }, 1500);
    }

    roundFailed() {
        this.gameState = 'failed';
        this.expressionOverlay.innerHTML = '&#x274C; Time\'s up!';
        this.expressionOverlay.style.color = '#ff6b6b';

        setTimeout(() => {
            this.round++;
            this.expressionOverlay.style.color = '#fff';
            this.nextRound();
        }, 1500);
    }

    endGame() {
        this.gameState = 'gameover';
        this.targetEmoji.innerHTML = '&#x1F3C6;';
        this.targetName.textContent = 'GAME OVER';
        this.timer.textContent = '';
        this.expressionOverlay.innerHTML = `Final Score: ${this.score} / ${this.totalRounds * 100}`;
        this.holdTimer.style.display = 'none';

        // Show restart option
        setTimeout(() => {
            this.expressionOverlay.innerHTML += '<br><br>Raise eyebrows to play again!';
        }, 2000);
    }

    handleEyebrowAction() {
        if (this.gameState === 'gameover') {
            this.startGame();
        }
    }

    updateScoreDisplay() {
        this.scoreDisplay.textContent = this.score;
    }
}

// Main initialization
async function initGame() {
    const loadingDiv = document.getElementById('loading');
    const gameArea = document.getElementById('gameArea');
    const statusText = document.getElementById('status');

    // Initialize face detector
    const faceDetector = new FaceDetector();
    const video = document.getElementById('webcam');
    const overlay = document.getElementById('overlay');

    // Initialize game
    const game = new ExpressionGame();

    // Load face detection model
    const modelLoaded = await faceDetector.init(video, overlay);
    if (!modelLoaded) {
        loadingDiv.innerHTML = '<p style="color: #ff6b6b;">Failed to load face detection model. Please refresh the page.</p>';
        return;
    }

    // Start camera
    const cameraStarted = await faceDetector.startCamera();
    if (!cameraStarted) {
        loadingDiv.innerHTML = '<p style="color: #ff6b6b;">Failed to access camera. Please allow camera permission.</p>';
        return;
    }

    // Show game area
    loadingDiv.style.display = 'none';
    gameArea.style.display = 'block';

    // Set up face detector callbacks
    faceDetector.onCalibrationProgress = (current, total) => {
        statusText.textContent = `Calibrating... ${current}/${total} - Keep a neutral face`;
    };

    faceDetector.onCalibrationComplete = () => {
        statusText.textContent = 'Ready! Match the expressions!';
        statusText.style.color = '#4a9';
    };

    faceDetector.onFaceDetected = (data) => {
        game.update(data);

        if (data.eyebrowAction) {
            game.handleEyebrowAction();
        }

        if (!data.calibrating) {
            statusText.textContent = `EAR: ${data.ear?.average.toFixed(3) || 'N/A'} | BAR: ${data.bar?.toFixed(3) || 'N/A'} | MAR: ${data.mar?.toFixed(3) || 'N/A'}`;
        }
    };

    faceDetector.onNoFace = () => {
        statusText.textContent = 'No face detected - look at camera!';
        statusText.style.color = '#ff6b6b';
    };

    // Game loop
    async function gameLoop() {
        await faceDetector.processFrame();
        requestAnimationFrame(gameLoop);
    }

    requestAnimationFrame(gameLoop);
}

// Start when page loads
window.addEventListener('load', initGame);
