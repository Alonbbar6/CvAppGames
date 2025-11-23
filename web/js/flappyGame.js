/**
 * Flappy Bird Game with Eyebrow Control
 */

class FlappyGame {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.width = canvas.width;
        this.height = canvas.height;

        // Game state
        this.gameState = 'waiting'; // waiting, playing, gameover
        this.score = 0;
        this.highScore = parseInt(localStorage.getItem('flappyHighScore') || '0');

        // Bird properties
        this.bird = {
            x: 50,
            y: this.height / 2,
            width: 34,
            height: 24,
            velocity: 0,
            gravity: 0.5,
            jumpStrength: -8,
            rotation: 0
        };

        // Pipe properties
        this.pipes = [];
        this.pipeWidth = 52;
        this.pipeGap = 120;
        this.pipeSpeed = 2;
        this.pipeSpawnInterval = 1500;
        this.lastPipeSpawn = 0;

        // Floor
        this.floorY = this.height - 56;
        this.floorX = 0;

        // Animation
        this.birdFrame = 0;
        this.frameCount = 0;

        // Colors
        this.colors = {
            sky: '#70c5ce',
            pipe: '#73bf2e',
            pipeEdge: '#558822',
            ground: '#ded895',
            groundDark: '#dab968'
        };

        this.init();
    }

    init() {
        // Keyboard controls
        document.addEventListener('keydown', (e) => {
            if (e.code === 'Space') {
                this.handleFlap();
            }
        });

        // Touch/click controls
        this.canvas.addEventListener('click', () => this.handleFlap());
    }

    handleFlap() {
        if (this.gameState === 'waiting') {
            this.startGame();
        } else if (this.gameState === 'playing') {
            this.flap();
        } else if (this.gameState === 'gameover') {
            this.resetGame();
        }
    }

    flap() {
        if (this.gameState === 'playing') {
            this.bird.velocity = this.bird.jumpStrength;
            this.bird.rotation = -25;
        }
    }

    startGame() {
        this.gameState = 'playing';
        this.bird.velocity = this.bird.jumpStrength;
    }

    resetGame() {
        this.gameState = 'waiting';
        this.score = 0;
        this.bird.y = this.height / 2;
        this.bird.velocity = 0;
        this.bird.rotation = 0;
        this.pipes = [];
        this.lastPipeSpawn = 0;
        this.updateScoreDisplay();
    }

    spawnPipe() {
        const minHeight = 50;
        const maxHeight = this.floorY - this.pipeGap - minHeight;
        const topHeight = minHeight + Math.random() * (maxHeight - minHeight);

        this.pipes.push({
            x: this.width,
            topHeight: topHeight,
            passed: false
        });
    }

    update(deltaTime) {
        this.frameCount++;

        if (this.gameState === 'waiting') {
            // Hovering animation
            this.bird.y = this.height / 2 + Math.sin(this.frameCount * 0.1) * 10;
            this.floorX = (this.floorX - 1) % 24;
            return;
        }

        if (this.gameState === 'gameover') {
            return;
        }

        // Update bird
        this.bird.velocity += this.bird.gravity;
        this.bird.y += this.bird.velocity;

        // Rotation based on velocity
        if (this.bird.velocity > 0) {
            this.bird.rotation = Math.min(this.bird.rotation + 3, 70);
        }

        // Spawn pipes
        if (Date.now() - this.lastPipeSpawn > this.pipeSpawnInterval) {
            this.spawnPipe();
            this.lastPipeSpawn = Date.now();
        }

        // Update pipes
        for (let i = this.pipes.length - 1; i >= 0; i--) {
            this.pipes[i].x -= this.pipeSpeed;

            // Score
            if (!this.pipes[i].passed && this.pipes[i].x + this.pipeWidth < this.bird.x) {
                this.pipes[i].passed = true;
                this.score++;
                this.updateScoreDisplay();
            }

            // Remove off-screen pipes
            if (this.pipes[i].x + this.pipeWidth < 0) {
                this.pipes.splice(i, 1);
            }
        }

        // Update floor
        this.floorX = (this.floorX - this.pipeSpeed) % 24;

        // Collision detection
        this.checkCollisions();
    }

    checkCollisions() {
        // Floor collision
        if (this.bird.y + this.bird.height > this.floorY) {
            this.gameOver();
            return;
        }

        // Ceiling collision
        if (this.bird.y < 0) {
            this.bird.y = 0;
            this.bird.velocity = 0;
        }

        // Pipe collision
        const birdBox = {
            x: this.bird.x + 3,
            y: this.bird.y + 3,
            width: this.bird.width - 6,
            height: this.bird.height - 6
        };

        for (const pipe of this.pipes) {
            // Top pipe
            if (this.checkBoxCollision(birdBox, {
                x: pipe.x,
                y: 0,
                width: this.pipeWidth,
                height: pipe.topHeight
            })) {
                this.gameOver();
                return;
            }

            // Bottom pipe
            if (this.checkBoxCollision(birdBox, {
                x: pipe.x,
                y: pipe.topHeight + this.pipeGap,
                width: this.pipeWidth,
                height: this.floorY - pipe.topHeight - this.pipeGap
            })) {
                this.gameOver();
                return;
            }
        }
    }

    checkBoxCollision(box1, box2) {
        return box1.x < box2.x + box2.width &&
               box1.x + box1.width > box2.x &&
               box1.y < box2.y + box2.height &&
               box1.y + box1.height > box2.y;
    }

    gameOver() {
        this.gameState = 'gameover';
        if (this.score > this.highScore) {
            this.highScore = this.score;
            localStorage.setItem('flappyHighScore', this.highScore.toString());
        }
    }

    updateScoreDisplay() {
        document.getElementById('score').textContent = this.score;
    }

    draw() {
        // Clear canvas
        this.ctx.fillStyle = this.colors.sky;
        this.ctx.fillRect(0, 0, this.width, this.height);

        // Draw pipes
        for (const pipe of this.pipes) {
            this.drawPipe(pipe);
        }

        // Draw floor
        this.drawFloor();

        // Draw bird
        this.drawBird();

        // Draw UI
        this.drawUI();
    }

    drawPipe(pipe) {
        const edgeHeight = 24;

        // Top pipe body
        this.ctx.fillStyle = this.colors.pipe;
        this.ctx.fillRect(pipe.x, 0, this.pipeWidth, pipe.topHeight - edgeHeight);

        // Top pipe edge
        this.ctx.fillStyle = this.colors.pipeEdge;
        this.ctx.fillRect(pipe.x - 2, pipe.topHeight - edgeHeight, this.pipeWidth + 4, edgeHeight);

        // Bottom pipe body
        const bottomY = pipe.topHeight + this.pipeGap;
        this.ctx.fillStyle = this.colors.pipe;
        this.ctx.fillRect(pipe.x, bottomY + edgeHeight, this.pipeWidth, this.floorY - bottomY - edgeHeight);

        // Bottom pipe edge
        this.ctx.fillStyle = this.colors.pipeEdge;
        this.ctx.fillRect(pipe.x - 2, bottomY, this.pipeWidth + 4, edgeHeight);

        // Highlights
        this.ctx.fillStyle = 'rgba(255,255,255,0.3)';
        this.ctx.fillRect(pipe.x + 3, 0, 5, pipe.topHeight - edgeHeight);
        this.ctx.fillRect(pipe.x + 3, bottomY + edgeHeight, 5, this.floorY - bottomY - edgeHeight);
    }

    drawFloor() {
        // Ground
        this.ctx.fillStyle = this.colors.ground;
        this.ctx.fillRect(0, this.floorY, this.width, this.height - this.floorY);

        // Ground pattern
        this.ctx.fillStyle = this.colors.groundDark;
        for (let x = this.floorX; x < this.width; x += 24) {
            this.ctx.fillRect(x, this.floorY, 12, 10);
        }
    }

    drawBird() {
        this.ctx.save();
        this.ctx.translate(this.bird.x + this.bird.width / 2, this.bird.y + this.bird.height / 2);
        this.ctx.rotate(this.bird.rotation * Math.PI / 180);

        // Bird body
        this.ctx.fillStyle = '#f8d347';
        this.ctx.beginPath();
        this.ctx.ellipse(0, 0, this.bird.width / 2, this.bird.height / 2, 0, 0, Math.PI * 2);
        this.ctx.fill();

        // Wing
        const wingY = this.gameState === 'playing' && this.frameCount % 10 < 5 ? -3 : 3;
        this.ctx.fillStyle = '#e8b830';
        this.ctx.beginPath();
        this.ctx.ellipse(-3, wingY, 10, 6, 0, 0, Math.PI * 2);
        this.ctx.fill();

        // Eye
        this.ctx.fillStyle = '#fff';
        this.ctx.beginPath();
        this.ctx.arc(8, -3, 7, 0, Math.PI * 2);
        this.ctx.fill();

        this.ctx.fillStyle = '#000';
        this.ctx.beginPath();
        this.ctx.arc(10, -3, 4, 0, Math.PI * 2);
        this.ctx.fill();

        // Beak
        this.ctx.fillStyle = '#e85e3a';
        this.ctx.beginPath();
        this.ctx.moveTo(this.bird.width / 2 - 5, 2);
        this.ctx.lineTo(this.bird.width / 2 + 8, 5);
        this.ctx.lineTo(this.bird.width / 2 - 5, 8);
        this.ctx.closePath();
        this.ctx.fill();

        this.ctx.restore();
    }

    drawUI() {
        if (this.gameState === 'waiting') {
            this.ctx.fillStyle = 'rgba(0,0,0,0.5)';
            this.ctx.fillRect(0, this.height / 2 - 40, this.width, 80);

            this.ctx.fillStyle = '#fff';
            this.ctx.font = '16px "Press Start 2P"';
            this.ctx.textAlign = 'center';
            this.ctx.fillText('RAISE EYEBROWS', this.width / 2, this.height / 2 - 10);
            this.ctx.fillText('TO START', this.width / 2, this.height / 2 + 15);
        }

        if (this.gameState === 'gameover') {
            this.ctx.fillStyle = 'rgba(0,0,0,0.7)';
            this.ctx.fillRect(0, 0, this.width, this.height);

            this.ctx.fillStyle = '#ff6b6b';
            this.ctx.font = '24px "Press Start 2P"';
            this.ctx.textAlign = 'center';
            this.ctx.fillText('GAME OVER', this.width / 2, this.height / 2 - 40);

            this.ctx.fillStyle = '#fff';
            this.ctx.font = '14px "Press Start 2P"';
            this.ctx.fillText(`Score: ${this.score}`, this.width / 2, this.height / 2);
            this.ctx.fillText(`Best: ${this.highScore}`, this.width / 2, this.height / 2 + 25);

            this.ctx.font = '10px "Press Start 2P"';
            this.ctx.fillStyle = '#7b2cbf';
            this.ctx.fillText('Raise eyebrows to restart', this.width / 2, this.height / 2 + 60);
        }

        // Score during gameplay
        if (this.gameState === 'playing') {
            this.ctx.fillStyle = '#fff';
            this.ctx.strokeStyle = '#000';
            this.ctx.lineWidth = 3;
            this.ctx.font = '28px "Press Start 2P"';
            this.ctx.textAlign = 'center';
            this.ctx.strokeText(this.score.toString(), this.width / 2, 50);
            this.ctx.fillText(this.score.toString(), this.width / 2, 50);
        }
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
    const gameCanvas = document.getElementById('gameCanvas');
    const game = new FlappyGame(gameCanvas);

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
        statusText.textContent = 'Ready! Raise your eyebrows to play!';
        statusText.style.color = '#4a9';
    };

    faceDetector.onFaceDetected = (data) => {
        if (!data.calibrating) {
            // Update status
            if (data.eyebrowsRaised) {
                statusText.textContent = 'EYEBROWS RAISED!';
                statusText.style.color = '#00d4ff';
            } else {
                statusText.textContent = `BAR: ${data.bar.toFixed(3)} / ${data.threshold.toFixed(3)}`;
                statusText.style.color = '#fff';
            }

            // Handle eyebrow action
            if (data.eyebrowAction) {
                game.handleFlap();
            }
        }
    };

    faceDetector.onNoFace = () => {
        statusText.textContent = 'No face detected - look at camera!';
        statusText.style.color = '#ff6b6b';
    };

    // Game loop
    let lastTime = 0;
    async function gameLoop(currentTime) {
        const deltaTime = currentTime - lastTime;
        lastTime = currentTime;

        // Process face detection
        await faceDetector.processFrame();

        // Update and draw game
        game.update(deltaTime);
        game.draw();

        requestAnimationFrame(gameLoop);
    }

    requestAnimationFrame(gameLoop);
}

// Start when page loads
window.addEventListener('load', initGame);
