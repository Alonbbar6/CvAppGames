/**
 * Tetris Game with Face Movement Control
 */

class TetrisGame {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');

        // Board dimensions
        this.cols = 10;
        this.rows = 20;
        this.blockSize = 24;

        // Board offset for side panel
        this.boardOffsetX = 10;
        this.boardOffsetY = 10;

        // Game state
        this.board = this.createBoard();
        this.score = 0;
        this.lines = 0;
        this.level = 1;
        this.gameOver = false;
        this.paused = false;

        // Current and next piece
        this.currentPiece = null;
        this.nextPiece = null;

        // Timing
        this.dropInterval = 1000;
        this.lastDrop = 0;
        this.lastMove = 0;
        this.moveCooldown = 150;

        // Colors for pieces
        this.colors = [
            null,
            '#00f0f0', // I - Cyan
            '#0000f0', // J - Blue
            '#f0a000', // L - Orange
            '#f0f000', // O - Yellow
            '#00f000', // S - Green
            '#a000f0', // T - Purple
            '#f00000', // Z - Red
        ];

        // Piece shapes
        this.shapes = [
            null,
            [[1,1,1,1]], // I
            [[2,0,0],[2,2,2]], // J
            [[0,0,3],[3,3,3]], // L
            [[4,4],[4,4]], // O
            [[0,5,5],[5,5,0]], // S
            [[0,6,0],[6,6,6]], // T
            [[7,7,0],[0,7,7]], // Z
        ];

        this.init();
    }

    init() {
        // Keyboard controls
        document.addEventListener('keydown', (e) => this.handleKeyDown(e));

        // Start game
        this.spawnPiece();
    }

    handleKeyDown(e) {
        if (this.gameOver) {
            if (e.code === 'Enter') {
                this.resetGame();
            }
            return;
        }

        switch (e.code) {
            case 'ArrowLeft':
            case 'KeyA':
                this.moveLeft();
                break;
            case 'ArrowRight':
            case 'KeyD':
                this.moveRight();
                break;
            case 'ArrowUp':
            case 'KeyW':
                this.rotate();
                break;
            case 'ArrowDown':
            case 'KeyS':
                this.softDrop();
                break;
            case 'Space':
                this.hardDrop();
                break;
        }
    }

    createBoard() {
        return Array.from({ length: this.rows }, () => Array(this.cols).fill(0));
    }

    spawnPiece() {
        if (this.nextPiece) {
            this.currentPiece = this.nextPiece;
        } else {
            this.currentPiece = this.createPiece();
        }
        this.nextPiece = this.createPiece();

        // Center the piece
        this.currentPiece.x = Math.floor((this.cols - this.currentPiece.shape[0].length) / 2);
        this.currentPiece.y = 0;

        // Check game over
        if (!this.isValidPosition(this.currentPiece.shape, this.currentPiece.x, this.currentPiece.y)) {
            this.gameOver = true;
        }
    }

    createPiece() {
        const type = Math.floor(Math.random() * 7) + 1;
        return {
            shape: this.shapes[type].map(row => [...row]),
            type: type,
            x: 0,
            y: 0
        };
    }

    isValidPosition(shape, offsetX, offsetY) {
        for (let y = 0; y < shape.length; y++) {
            for (let x = 0; x < shape[y].length; x++) {
                if (shape[y][x]) {
                    const newX = x + offsetX;
                    const newY = y + offsetY;

                    if (newX < 0 || newX >= this.cols || newY >= this.rows) {
                        return false;
                    }
                    if (newY >= 0 && this.board[newY][newX]) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    moveLeft() {
        if (this.isValidPosition(this.currentPiece.shape, this.currentPiece.x - 1, this.currentPiece.y)) {
            this.currentPiece.x--;
        }
    }

    moveRight() {
        if (this.isValidPosition(this.currentPiece.shape, this.currentPiece.x + 1, this.currentPiece.y)) {
            this.currentPiece.x++;
        }
    }

    rotate() {
        const rotated = this.rotateMatrix(this.currentPiece.shape);
        if (this.isValidPosition(rotated, this.currentPiece.x, this.currentPiece.y)) {
            this.currentPiece.shape = rotated;
        }
    }

    rotateMatrix(matrix) {
        const rows = matrix.length;
        const cols = matrix[0].length;
        const rotated = Array.from({ length: cols }, () => Array(rows).fill(0));
        for (let y = 0; y < rows; y++) {
            for (let x = 0; x < cols; x++) {
                rotated[x][rows - 1 - y] = matrix[y][x];
            }
        }
        return rotated;
    }

    softDrop() {
        if (this.isValidPosition(this.currentPiece.shape, this.currentPiece.x, this.currentPiece.y + 1)) {
            this.currentPiece.y++;
            this.score += 1;
        }
    }

    hardDrop() {
        while (this.isValidPosition(this.currentPiece.shape, this.currentPiece.x, this.currentPiece.y + 1)) {
            this.currentPiece.y++;
            this.score += 2;
        }
        this.lockPiece();
    }

    lockPiece() {
        const shape = this.currentPiece.shape;
        const offsetX = this.currentPiece.x;
        const offsetY = this.currentPiece.y;

        for (let y = 0; y < shape.length; y++) {
            for (let x = 0; x < shape[y].length; x++) {
                if (shape[y][x]) {
                    const boardY = y + offsetY;
                    if (boardY >= 0) {
                        this.board[boardY][x + offsetX] = shape[y][x];
                    }
                }
            }
        }

        this.clearLines();
        this.spawnPiece();
    }

    clearLines() {
        let linesCleared = 0;

        for (let y = this.rows - 1; y >= 0; y--) {
            if (this.board[y].every(cell => cell !== 0)) {
                this.board.splice(y, 1);
                this.board.unshift(Array(this.cols).fill(0));
                linesCleared++;
                y++; // Check same row again
            }
        }

        if (linesCleared > 0) {
            const points = [0, 100, 300, 500, 800];
            this.score += points[linesCleared] * this.level;
            this.lines += linesCleared;
            this.level = Math.floor(this.lines / 10) + 1;
            this.dropInterval = Math.max(100, 1000 - (this.level - 1) * 100);
            this.updateScoreDisplay();
        }
    }

    updateScoreDisplay() {
        document.getElementById('score').textContent = this.score;
    }

    update(currentTime) {
        if (this.gameOver || this.paused) return;

        // Auto drop
        if (currentTime - this.lastDrop > this.dropInterval) {
            if (this.isValidPosition(this.currentPiece.shape, this.currentPiece.x, this.currentPiece.y + 1)) {
                this.currentPiece.y++;
            } else {
                this.lockPiece();
            }
            this.lastDrop = currentTime;
        }
    }

    // Face control methods
    handleFaceMovement(movement, currentTime) {
        if (this.gameOver) return;

        if (currentTime - this.lastMove > this.moveCooldown) {
            if (movement === 'left') {
                this.moveLeft();
                this.lastMove = currentTime;
            } else if (movement === 'right') {
                this.moveRight();
                this.lastMove = currentTime;
            }
        }
    }

    handleEyebrowAction() {
        if (this.gameOver) return;
        this.rotate();
    }

    resetGame() {
        this.board = this.createBoard();
        this.score = 0;
        this.lines = 0;
        this.level = 1;
        this.gameOver = false;
        this.dropInterval = 1000;
        this.currentPiece = null;
        this.nextPiece = null;
        this.spawnPiece();
        this.updateScoreDisplay();
    }

    draw() {
        // Clear canvas
        this.ctx.fillStyle = '#1a1a2e';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        // Draw board background
        this.ctx.fillStyle = '#0f0f1a';
        this.ctx.fillRect(
            this.boardOffsetX,
            this.boardOffsetY,
            this.cols * this.blockSize,
            this.rows * this.blockSize
        );

        // Draw grid
        this.ctx.strokeStyle = '#2a2a3e';
        for (let x = 0; x <= this.cols; x++) {
            this.ctx.beginPath();
            this.ctx.moveTo(this.boardOffsetX + x * this.blockSize, this.boardOffsetY);
            this.ctx.lineTo(this.boardOffsetX + x * this.blockSize, this.boardOffsetY + this.rows * this.blockSize);
            this.ctx.stroke();
        }
        for (let y = 0; y <= this.rows; y++) {
            this.ctx.beginPath();
            this.ctx.moveTo(this.boardOffsetX, this.boardOffsetY + y * this.blockSize);
            this.ctx.lineTo(this.boardOffsetX + this.cols * this.blockSize, this.boardOffsetY + y * this.blockSize);
            this.ctx.stroke();
        }

        // Draw locked pieces
        for (let y = 0; y < this.rows; y++) {
            for (let x = 0; x < this.cols; x++) {
                if (this.board[y][x]) {
                    this.drawBlock(x, y, this.board[y][x]);
                }
            }
        }

        // Draw current piece
        if (this.currentPiece && !this.gameOver) {
            const shape = this.currentPiece.shape;
            for (let y = 0; y < shape.length; y++) {
                for (let x = 0; x < shape[y].length; x++) {
                    if (shape[y][x]) {
                        this.drawBlock(
                            x + this.currentPiece.x,
                            y + this.currentPiece.y,
                            shape[y][x]
                        );
                    }
                }
            }
        }

        // Draw side panel
        this.drawSidePanel();

        // Draw game over
        if (this.gameOver) {
            this.drawGameOver();
        }
    }

    drawBlock(x, y, type) {
        const px = this.boardOffsetX + x * this.blockSize;
        const py = this.boardOffsetY + y * this.blockSize;

        // Main color
        this.ctx.fillStyle = this.colors[type];
        this.ctx.fillRect(px + 1, py + 1, this.blockSize - 2, this.blockSize - 2);

        // Highlight
        this.ctx.fillStyle = 'rgba(255,255,255,0.3)';
        this.ctx.fillRect(px + 1, py + 1, this.blockSize - 2, 3);
        this.ctx.fillRect(px + 1, py + 1, 3, this.blockSize - 2);

        // Shadow
        this.ctx.fillStyle = 'rgba(0,0,0,0.3)';
        this.ctx.fillRect(px + this.blockSize - 4, py + 1, 3, this.blockSize - 2);
        this.ctx.fillRect(px + 1, py + this.blockSize - 4, this.blockSize - 2, 3);
    }

    drawSidePanel() {
        const panelX = this.boardOffsetX + this.cols * this.blockSize + 15;

        // Score
        this.ctx.fillStyle = '#00d4ff';
        this.ctx.font = '12px "Press Start 2P"';
        this.ctx.fillText('SCORE', panelX, 30);
        this.ctx.fillStyle = '#fff';
        this.ctx.font = '14px "Press Start 2P"';
        this.ctx.fillText(this.score.toString(), panelX, 55);

        // Lines
        this.ctx.fillStyle = '#00d4ff';
        this.ctx.font = '12px "Press Start 2P"';
        this.ctx.fillText('LINES', panelX, 90);
        this.ctx.fillStyle = '#fff';
        this.ctx.font = '14px "Press Start 2P"';
        this.ctx.fillText(this.lines.toString(), panelX, 115);

        // Level
        this.ctx.fillStyle = '#00d4ff';
        this.ctx.font = '12px "Press Start 2P"';
        this.ctx.fillText('LEVEL', panelX, 150);
        this.ctx.fillStyle = '#fff';
        this.ctx.font = '14px "Press Start 2P"';
        this.ctx.fillText(this.level.toString(), panelX, 175);

        // Next piece
        this.ctx.fillStyle = '#00d4ff';
        this.ctx.font = '12px "Press Start 2P"';
        this.ctx.fillText('NEXT', panelX, 220);

        if (this.nextPiece) {
            const shape = this.nextPiece.shape;
            const previewX = panelX + 10;
            const previewY = 240;
            const previewSize = 18;

            for (let y = 0; y < shape.length; y++) {
                for (let x = 0; x < shape[y].length; x++) {
                    if (shape[y][x]) {
                        this.ctx.fillStyle = this.colors[shape[y][x]];
                        this.ctx.fillRect(
                            previewX + x * previewSize,
                            previewY + y * previewSize,
                            previewSize - 2,
                            previewSize - 2
                        );
                    }
                }
            }
        }
    }

    drawGameOver() {
        this.ctx.fillStyle = 'rgba(0,0,0,0.8)';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        this.ctx.fillStyle = '#ff6b6b';
        this.ctx.font = '20px "Press Start 2P"';
        this.ctx.textAlign = 'center';
        this.ctx.fillText('GAME', this.canvas.width / 2, this.canvas.height / 2 - 30);
        this.ctx.fillText('OVER', this.canvas.width / 2, this.canvas.height / 2);

        this.ctx.fillStyle = '#fff';
        this.ctx.font = '12px "Press Start 2P"';
        this.ctx.fillText(`Score: ${this.score}`, this.canvas.width / 2, this.canvas.height / 2 + 40);

        this.ctx.fillStyle = '#7b2cbf';
        this.ctx.font = '10px "Press Start 2P"';
        this.ctx.fillText('Press ENTER', this.canvas.width / 2, this.canvas.height / 2 + 80);
        this.ctx.fillText('to restart', this.canvas.width / 2, this.canvas.height / 2 + 95);

        this.ctx.textAlign = 'left';
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
    const game = new TetrisGame(gameCanvas);

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
        statusText.textContent = 'Ready! Move head LEFT/RIGHT, raise eyebrows to ROTATE';
        statusText.style.color = '#4a9';
    };

    faceDetector.onFaceDetected = (data) => {
        if (!data.calibrating) {
            // Update status based on movement
            let statusMsg = '';
            if (data.movement === 'left') {
                statusMsg = '< LEFT';
                statusText.style.color = '#00d4ff';
            } else if (data.movement === 'right') {
                statusMsg = 'RIGHT >';
                statusText.style.color = '#00d4ff';
            } else {
                statusMsg = 'CENTER';
                statusText.style.color = '#4a9';
            }

            if (data.eyebrowsRaised) {
                statusMsg += ' | ROTATE!';
                statusText.style.color = '#7b2cbf';
            }

            statusText.textContent = statusMsg;

            // Handle face controls
            const currentTime = Date.now();
            game.handleFaceMovement(data.movement, currentTime);

            if (data.eyebrowAction) {
                game.handleEyebrowAction();
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
        // Process face detection
        await faceDetector.processFrame();

        // Update and draw game
        game.update(currentTime);
        game.draw();

        requestAnimationFrame(gameLoop);
    }

    requestAnimationFrame(gameLoop);
}

// Start when page loads
window.addEventListener('load', initGame);
