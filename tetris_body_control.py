"""
Tetris controlled by Body Movement!
Lean left/right to move pieces, raise eyebrows to rotate.
Uses OpenCV face detection for tracking.
"""

import cv2
import numpy as np
import pygame
import random
import time
import os

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Model file paths
MODEL_PATH = os.path.join(SCRIPT_DIR, 'model/res10_300x300_ssd_iter_140000.caffemodel')
CONFIG_PATH = os.path.join(SCRIPT_DIR, 'model/deploy.prototxt')
LBF_MODEL = os.path.join(SCRIPT_DIR, 'model/lbfmodel.yaml')

print("Loading face detection models...")
net = cv2.dnn.readNetFromCaffe(CONFIG_PATH, MODEL_PATH)
landmarkDetector = cv2.face.createFacemarkLBF()
landmarkDetector.loadModel(LBF_MODEL)
print("Models loaded!")

#------------------------------------------------------------------------------
# Tetris Game Constants
#------------------------------------------------------------------------------

# Colors (RGB)
COLORS = [
    (0, 0, 0),        # Empty
    (0, 255, 255),    # I - Cyan
    (0, 0, 255),      # J - Blue
    (255, 165, 0),    # L - Orange
    (255, 255, 0),    # O - Yellow
    (0, 255, 0),      # S - Green
    (128, 0, 128),    # T - Purple
    (255, 0, 0),      # Z - Red
]

# Tetromino shapes
SHAPES = [
    [[1, 1, 1, 1]],  # I
    [[2, 0, 0], [2, 2, 2]],  # J
    [[0, 0, 3], [3, 3, 3]],  # L
    [[4, 4], [4, 4]],  # O
    [[0, 5, 5], [5, 5, 0]],  # S
    [[0, 6, 0], [6, 6, 6]],  # T
    [[7, 7, 0], [0, 7, 7]],  # Z
]

BLOCK_SIZE = 30
BOARD_WIDTH = 10
BOARD_HEIGHT = 20

#------------------------------------------------------------------------------
# Helper Functions
#------------------------------------------------------------------------------

def calculate_distance(A, B):
    return ((A[0] - B[0])**2 + (A[1] - B[1])**2)**0.5

def detect_faces(image, detection_threshold=0.70):
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123])
    net.setInput(blob)
    detections = net.forward()
    faces = []
    img_h, img_w = image.shape[:2]

    for detection in detections[0][0]:
        if detection[2] >= detection_threshold:
            left = detection[3] * img_w
            top = detection[4] * img_h
            right = detection[5] * img_w
            bottom = detection[6] * img_h
            face_w = right - left
            face_h = bottom - top
            faces.append((left, top, face_w, face_h))

    return np.array(faces).astype(int) if faces else np.array([])

def get_eyebrow_aspect_ratio(landmarks):
    # Right eyebrow center to right eye top
    right_height = calculate_distance(landmarks[19], landmarks[37])
    # Left eyebrow center to left eye top
    left_height = calculate_distance(landmarks[24], landmarks[44])

    # Eyebrow widths
    right_width = calculate_distance(landmarks[17], landmarks[21])
    left_width = calculate_distance(landmarks[22], landmarks[26])

    right_bar = right_height / right_width
    left_bar = left_height / left_width

    return (right_bar + left_bar) / 2

#------------------------------------------------------------------------------
# Body Movement Detector (using face position)
#------------------------------------------------------------------------------

class BodyPoseDetector:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        ret, frame = self.cap.read()
        if ret:
            self.frame_h, self.frame_w = frame.shape[:2]
        else:
            self.frame_h, self.frame_w = 480, 640

        # Movement zones based on face position
        self.center_x = self.frame_w // 2
        self.left_zone = self.frame_w * 0.35
        self.right_zone = self.frame_w * 0.65

        # Calibration for eyebrow detection
        self.calibrated = False
        self.calibration_frames = 0
        self.calibration_needed = 30
        self.baseline_bar = 0
        self.sum_bar = 0
        self.raise_threshold = 0

        # Action cooldowns
        self.last_move_time = 0
        self.move_cooldown = 0.15  # 150ms between moves
        self.last_rotate_time = 0
        self.rotate_cooldown = 0.4  # 400ms between rotates

        # State tracking
        self.eyebrows_were_raised = False

        # Face tracking persistence
        self.last_known_face = None
        self.face_lost_frames = 0
        self.max_face_lost_frames = 10

    def process_frame(self):
        """Process frame and return movement commands"""
        ret, frame = self.cap.read()
        if not ret:
            return None, None

        frame = cv2.flip(frame, 1)
        action = None
        current_time = time.time()

        # Detect faces
        faces = detect_faces(frame, detection_threshold=0.70)

        primary_face = None
        if len(faces) > 0:
            # Get largest face
            max_area = 0
            for face in faces:
                x, y, w, h = face
                if w * h > max_area:
                    max_area = w * h
                    primary_face = face
            if primary_face is not None:
                self.last_known_face = primary_face.copy()
                self.face_lost_frames = 0
        elif self.last_known_face is not None and self.face_lost_frames < self.max_face_lost_frames:
            primary_face = self.last_known_face
            self.face_lost_frames += 1

        if primary_face is not None:
            x, y, w, h = primary_face
            face_center_x = x + w // 2
            face_center_y = y + h // 2

            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.circle(frame, (face_center_x, face_center_y), 5, (0, 255, 0), -1)

            # Detect landmarks for eyebrow detection
            retval, landmarksList = landmarkDetector.fit(frame, np.expand_dims(primary_face, 0))

            if retval:
                landmarks = landmarksList[0][0]

                # Draw eyebrow landmarks
                for i in range(17, 27):
                    cv2.circle(frame, tuple(landmarks[i].astype('int')), 2, (255, 0, 0), -1)

                bar = get_eyebrow_aspect_ratio(landmarks)

                if not self.calibrated:
                    self.calibrate(bar)
                    cv2.putText(frame, f"Calibrating... {self.calibration_frames}/{self.calibration_needed}",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                    cv2.putText(frame, "Look at camera with neutral face",
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                else:
                    # Draw zones
                    cv2.line(frame, (int(self.left_zone), 0), (int(self.left_zone), self.frame_h), (255, 255, 0), 2)
                    cv2.line(frame, (int(self.right_zone), 0), (int(self.right_zone), self.frame_h), (255, 255, 0), 2)

                    # Check for eyebrow raise (rotate)
                    eyebrows_raised = bar > self.raise_threshold

                    if eyebrows_raised and not self.eyebrows_were_raised:
                        if current_time - self.last_rotate_time > self.rotate_cooldown:
                            action = 'ROTATE'
                            self.last_rotate_time = current_time
                            cv2.putText(frame, "ROTATE!", (self.frame_w//2 - 60, 50),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 3)

                    self.eyebrows_were_raised = eyebrows_raised

                    # Check for left/right movement based on face position
                    if face_center_x < self.left_zone:
                        if current_time - self.last_move_time > self.move_cooldown:
                            action = 'LEFT'
                            self.last_move_time = current_time
                        cv2.putText(frame, "< LEFT", (10, 100),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                    elif face_center_x > self.right_zone:
                        if current_time - self.last_move_time > self.move_cooldown:
                            action = 'RIGHT'
                            self.last_move_time = current_time
                        cv2.putText(frame, "RIGHT >", (self.frame_w - 150, 100),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                    else:
                        cv2.putText(frame, "CENTER", (self.frame_w//2 - 50, 100),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

                    # Show BAR value
                    color = (0, 255, 0) if eyebrows_raised else (255, 255, 255)
                    cv2.putText(frame, f"BAR: {bar:.3f} / {self.raise_threshold:.3f}",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    # Show status
                    cv2.putText(frame, "Move HEAD left/right | Raise EYEBROWS to rotate",
                               (10, self.frame_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            cv2.putText(frame, "No face detected - look at camera!", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow('Body Control - Tetris', frame)
        cv2.waitKey(1)

        return frame, action

    def calibrate(self, bar):
        if self.calibration_frames < self.calibration_needed:
            self.calibration_frames += 1
            self.sum_bar += bar
        else:
            self.baseline_bar = self.sum_bar / self.calibration_needed
            self.raise_threshold = self.baseline_bar * 1.12  # 12% above baseline
            self.calibrated = True
            print(f"\nCalibration complete!")
            print(f"Baseline BAR: {self.baseline_bar:.3f}")
            print(f"Raise threshold: {self.raise_threshold:.3f}")
            print("\n>>> MOVE HEAD left/right, RAISE EYEBROWS to rotate! <<<\n")

    def stop(self):
        self.cap.release()
        cv2.destroyAllWindows()

#------------------------------------------------------------------------------
# Tetris Game
#------------------------------------------------------------------------------

class Tetromino:
    def __init__(self, shape_idx=None):
        if shape_idx is None:
            shape_idx = random.randint(0, len(SHAPES) - 1)
        self.shape = [row[:] for row in SHAPES[shape_idx]]  # Deep copy
        self.x = BOARD_WIDTH // 2 - len(self.shape[0]) // 2
        self.y = 0

    def rotate(self):
        # Transpose and reverse rows for 90-degree clockwise rotation
        rows = len(self.shape)
        cols = len(self.shape[0])
        rotated = [[self.shape[rows - 1 - j][i] for j in range(rows)] for i in range(cols)]
        return rotated

class TetrisGame:
    def __init__(self, body_detector):
        self.body_detector = body_detector

        pygame.init()
        pygame.display.set_caption("Tetris - Body Control")

        # Game dimensions
        self.screen_width = BOARD_WIDTH * BLOCK_SIZE + 200  # Extra space for info
        self.screen_height = BOARD_HEIGHT * BLOCK_SIZE
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()

        self.reset_game()

    def reset_game(self):
        self.board = [[0] * BOARD_WIDTH for _ in range(BOARD_HEIGHT)]
        self.current_piece = Tetromino()
        self.next_piece = Tetromino()
        self.score = 0
        self.lines = 0
        self.level = 1
        self.game_over = False
        self.last_drop_time = time.time()
        self.drop_interval = 1.0  # Seconds between drops

    def valid_position(self, shape, offset_x, offset_y):
        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell:
                    new_x = x + offset_x
                    new_y = y + offset_y
                    if new_x < 0 or new_x >= BOARD_WIDTH:
                        return False
                    if new_y >= BOARD_HEIGHT:
                        return False
                    if new_y >= 0 and self.board[new_y][new_x]:
                        return False
        return True

    def lock_piece(self):
        for y, row in enumerate(self.current_piece.shape):
            for x, cell in enumerate(row):
                if cell:
                    board_y = y + self.current_piece.y
                    board_x = x + self.current_piece.x
                    if board_y >= 0:
                        self.board[board_y][board_x] = cell

        # Check for completed lines
        lines_cleared = 0
        y = BOARD_HEIGHT - 1
        while y >= 0:
            if all(self.board[y]):
                lines_cleared += 1
                del self.board[y]
                self.board.insert(0, [0] * BOARD_WIDTH)
            else:
                y -= 1

        if lines_cleared > 0:
            self.lines += lines_cleared
            self.score += lines_cleared * 100 * self.level
            self.level = self.lines // 10 + 1
            self.drop_interval = max(0.1, 1.0 - (self.level - 1) * 0.1)

        # Spawn new piece
        self.current_piece = self.next_piece
        self.next_piece = Tetromino()

        # Check game over
        if not self.valid_position(self.current_piece.shape,
                                   self.current_piece.x,
                                   self.current_piece.y):
            self.game_over = True

    def move_left(self):
        if self.valid_position(self.current_piece.shape,
                               self.current_piece.x - 1,
                               self.current_piece.y):
            self.current_piece.x -= 1

    def move_right(self):
        if self.valid_position(self.current_piece.shape,
                               self.current_piece.x + 1,
                               self.current_piece.y):
            self.current_piece.x += 1

    def rotate_piece(self):
        rotated = self.current_piece.rotate()
        if self.valid_position(rotated,
                               self.current_piece.x,
                               self.current_piece.y):
            self.current_piece.shape = rotated

    def drop(self):
        if self.valid_position(self.current_piece.shape,
                               self.current_piece.x,
                               self.current_piece.y + 1):
            self.current_piece.y += 1
        else:
            self.lock_piece()

    def hard_drop(self):
        while self.valid_position(self.current_piece.shape,
                                  self.current_piece.x,
                                  self.current_piece.y + 1):
            self.current_piece.y += 1
        self.lock_piece()

    def draw_block(self, x, y, color_idx, surface=None):
        if surface is None:
            surface = self.screen
        color = COLORS[color_idx]
        rect = pygame.Rect(x * BLOCK_SIZE, y * BLOCK_SIZE,
                          BLOCK_SIZE - 1, BLOCK_SIZE - 1)
        pygame.draw.rect(surface, color, rect)
        # Add highlight
        if color_idx > 0:
            highlight = pygame.Rect(x * BLOCK_SIZE, y * BLOCK_SIZE,
                                   BLOCK_SIZE - 1, 3)
            pygame.draw.rect(surface, tuple(min(255, c + 50) for c in color), highlight)

    def draw(self):
        self.screen.fill((20, 20, 20))

        # Draw board background
        pygame.draw.rect(self.screen, (40, 40, 40),
                        (0, 0, BOARD_WIDTH * BLOCK_SIZE, BOARD_HEIGHT * BLOCK_SIZE))

        # Draw grid
        for x in range(BOARD_WIDTH + 1):
            pygame.draw.line(self.screen, (60, 60, 60),
                           (x * BLOCK_SIZE, 0),
                           (x * BLOCK_SIZE, BOARD_HEIGHT * BLOCK_SIZE))
        for y in range(BOARD_HEIGHT + 1):
            pygame.draw.line(self.screen, (60, 60, 60),
                           (0, y * BLOCK_SIZE),
                           (BOARD_WIDTH * BLOCK_SIZE, y * BLOCK_SIZE))

        # Draw locked pieces
        for y, row in enumerate(self.board):
            for x, cell in enumerate(row):
                if cell:
                    self.draw_block(x, y, cell)

        # Draw current piece
        if not self.game_over:
            for y, row in enumerate(self.current_piece.shape):
                for x, cell in enumerate(row):
                    if cell:
                        self.draw_block(x + self.current_piece.x,
                                       y + self.current_piece.y, cell)

        # Draw info panel
        info_x = BOARD_WIDTH * BLOCK_SIZE + 20
        font = pygame.font.Font(None, 36)
        small_font = pygame.font.Font(None, 24)

        # Score
        score_text = font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (info_x, 20))

        # Lines
        lines_text = font.render(f"Lines: {self.lines}", True, (255, 255, 255))
        self.screen.blit(lines_text, (info_x, 60))

        # Level
        level_text = font.render(f"Level: {self.level}", True, (255, 255, 255))
        self.screen.blit(level_text, (info_x, 100))

        # Next piece
        next_text = font.render("Next:", True, (255, 255, 255))
        self.screen.blit(next_text, (info_x, 160))

        # Draw next piece preview
        for y, row in enumerate(self.next_piece.shape):
            for x, cell in enumerate(row):
                if cell:
                    px = info_x // BLOCK_SIZE + x
                    py = 7 + y
                    self.draw_block(px, py, cell)

        # Controls info
        ctrl1 = small_font.render("BODY CONTROLS:", True, (255, 255, 0))
        ctrl2 = small_font.render("Lean LEFT/RIGHT", True, (200, 200, 200))
        ctrl3 = small_font.render("Raise ARM = Rotate", True, (200, 200, 200))
        ctrl4 = small_font.render("KEYBOARD:", True, (255, 255, 0))
        ctrl5 = small_font.render("Arrows/WASD", True, (200, 200, 200))
        ctrl6 = small_font.render("Space = Drop", True, (200, 200, 200))

        self.screen.blit(ctrl1, (info_x, 350))
        self.screen.blit(ctrl2, (info_x, 375))
        self.screen.blit(ctrl3, (info_x, 395))
        self.screen.blit(ctrl4, (info_x, 430))
        self.screen.blit(ctrl5, (info_x, 455))
        self.screen.blit(ctrl6, (info_x, 475))

        if self.game_over:
            # Game over overlay
            overlay = pygame.Surface((self.screen_width, self.screen_height))
            overlay.set_alpha(180)
            overlay.fill((0, 0, 0))
            self.screen.blit(overlay, (0, 0))

            go_font = pygame.font.Font(None, 72)
            go_text = go_font.render("GAME OVER", True, (255, 0, 0))
            go_rect = go_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2 - 30))
            self.screen.blit(go_text, go_rect)

            restart_text = font.render("Press ENTER to restart", True, (255, 255, 255))
            restart_rect = restart_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2 + 30))
            self.screen.blit(restart_text, restart_rect)

        pygame.display.flip()

    def run(self):
        running = True

        while running:
            # Process body detection
            frame, action = self.body_detector.process_frame()

            # Handle body control actions
            if action and not self.game_over:
                if action == 'LEFT':
                    self.move_left()
                elif action == 'RIGHT':
                    self.move_right()
                elif action == 'ROTATE':
                    self.rotate_piece()

            # Handle keyboard events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif self.game_over:
                        if event.key == pygame.K_RETURN:
                            self.reset_game()
                    else:
                        if event.key in (pygame.K_LEFT, pygame.K_a):
                            self.move_left()
                        elif event.key in (pygame.K_RIGHT, pygame.K_d):
                            self.move_right()
                        elif event.key in (pygame.K_UP, pygame.K_w):
                            self.rotate_piece()
                        elif event.key in (pygame.K_DOWN, pygame.K_s):
                            self.drop()
                        elif event.key == pygame.K_SPACE:
                            self.hard_drop()

            # Auto drop
            if not self.game_over:
                current_time = time.time()
                if current_time - self.last_drop_time > self.drop_interval:
                    self.drop()
                    self.last_drop_time = current_time

            self.draw()
            self.clock.tick(60)

        self.body_detector.stop()
        pygame.quit()

#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------

def main():
    print("=" * 50)
    print("  TETRIS - FACE/BODY CONTROL")
    print("=" * 50)
    print("\nINSTRUCTIONS:")
    print("1. Look at camera with a neutral face during calibration")
    print("2. MOVE YOUR HEAD left/right to move pieces")
    print("3. RAISE YOUR EYEBROWS to ROTATE pieces")
    print("4. Use keyboard as backup (arrows/WASD, Space=drop)")
    print("5. Press ESC to quit")
    print("\n" + "=" * 50 + "\n")

    # Create body detector
    body_detector = BodyPoseDetector()

    # Start game
    game = TetrisGame(body_detector)

    try:
        game.run()
    except KeyboardInterrupt:
        pass
    finally:
        body_detector.stop()

if __name__ == "__main__":
    main()
