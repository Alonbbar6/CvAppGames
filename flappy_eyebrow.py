"""
Flappy Bird controlled by Eyebrow Raises!
Raise your eyebrows to make the bird jump.
"""

import asyncio
import sys
import os
import cv2
import numpy as np
import time

import pygame
from pygame.locals import K_ESCAPE, K_SPACE, K_UP, KEYDOWN, QUIT

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Add FlapPyBird to path
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'FlapPyBird'))

# Change to FlapPyBird directory for assets
os.chdir(os.path.join(SCRIPT_DIR, 'FlapPyBird'))

from src.entities import (
    Background,
    Floor,
    GameOver,
    Pipes,
    Player,
    PlayerMode,
    Score,
    WelcomeMessage,
)
from src.utils import GameConfig, Images, Sounds, Window

#------------------------------------------------------------------------------
# Face Detection Setup
#------------------------------------------------------------------------------

# Model file paths (use absolute paths)
MODEL_PATH = os.path.join(SCRIPT_DIR, 'model/res10_300x300_ssd_iter_140000.caffemodel')
CONFIG_PATH = os.path.join(SCRIPT_DIR, 'model/deploy.prototxt')
LBF_MODEL = os.path.join(SCRIPT_DIR, 'model/lbfmodel.yaml')

print("Loading face detection models...")
# Create face detector
net = cv2.dnn.readNetFromCaffe(CONFIG_PATH, MODEL_PATH)

# Create landmark detector
landmarkDetector = cv2.face.createFacemarkLBF()
landmarkDetector.loadModel(LBF_MODEL)
print("Models loaded!")

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

def get_primary_face(faces, frame_h, frame_w):
    primary_face = None
    max_height = 0

    for face in faces:
        x1, y1, w, h = face
        x2, y2 = x1 + w, y1 + h

        if x1 < 0 or y1 < 0 or x2 > frame_w or y2 > frame_h:
            continue

        if h > max_height:
            max_height = h
            primary_face = face

    return primary_face

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
# Face Detector Class
#------------------------------------------------------------------------------

class FaceDetector:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.eyebrow_raised = False
        self.eyebrow_was_raised = False
        self.flap_triggered = False

        # Calibration
        self.calibrated = False
        self.calibration_frames = 0
        self.calibration_needed = 30
        self.sum_bar = 0
        self.baseline_bar = 0
        self.raise_threshold = 0

        # Smoothing - keep last few BAR values
        self.bar_history = []
        self.history_size = 5

        # Keep raised state for a few frames even if detection is lost
        self.raised_cooldown = 0
        self.cooldown_frames = 3

        # Face tracking persistence - keep last known face when detection fails
        self.last_known_face = None
        self.face_lost_frames = 0
        self.max_face_lost_frames = 10  # Use last known face for up to 10 frames

        # Frame info
        ret, frame = self.cap.read()
        if ret:
            self.frame_h, self.frame_w = frame.shape[:2]
        else:
            self.frame_h, self.frame_w = 480, 640

    def process_frame(self):
        """Process one frame and return if a flap should happen"""
        ret, frame = self.cap.read()
        if not ret:
            return False

        # Flip for mirror effect
        frame = cv2.flip(frame, 1)

        # Detect faces with lower threshold to handle raised eyebrows
        faces = detect_faces(frame, detection_threshold=0.70)

        primary_face = None
        if len(faces) > 0:
            primary_face = get_primary_face(faces, self.frame_h, self.frame_w)
            if primary_face is not None:
                self.last_known_face = primary_face.copy()
                self.face_lost_frames = 0
        elif self.last_known_face is not None and self.face_lost_frames < self.max_face_lost_frames:
            # Use last known face position when detection fails temporarily
            primary_face = self.last_known_face
            self.face_lost_frames += 1
            cv2.putText(frame, f"Using cached face ({self.max_face_lost_frames - self.face_lost_frames} frames left)",
                       (10, self.frame_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

        if primary_face is not None:
            # Draw face rectangle
            x, y, w, h = primary_face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Detect landmarks
            retval, landmarksList = landmarkDetector.fit(frame, np.expand_dims(primary_face, 0))

            if retval:
                landmarks = landmarksList[0][0]

                # Draw eyebrow landmarks
                for i in range(17, 27):
                    cv2.circle(frame, tuple(landmarks[i].astype('int')), 2, (255, 0, 0), -1)

                # Calculate BAR
                bar = get_eyebrow_aspect_ratio(landmarks)

                if not self.calibrated:
                    self.calibrate(bar)
                    cv2.putText(frame, f"Calibrating... {self.calibration_frames}/{self.calibration_needed}",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                    cv2.putText(frame, "Keep a neutral face",
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                else:
                    # Check for eyebrow raise with hysteresis
                    # Must go ABOVE raise_threshold to trigger
                    # Must go BELOW reset_threshold before can trigger again
                    reset_threshold = self.baseline_bar * 1.05  # 5% above baseline to reset

                    if bar > self.raise_threshold:
                        # Eyebrows are raised
                        if not self.eyebrow_was_raised and self.raised_cooldown == 0:
                            # First frame of raise - trigger flap!
                            self.flap_triggered = True
                            self.raised_cooldown = 10  # Wait 10 frames before next flap
                            print(f"FLAP triggered! BAR: {bar:.3f}")
                        self.eyebrow_raised = True
                        self.eyebrow_was_raised = True
                        cv2.putText(frame, "FLAP!", (self.frame_w//2 - 60, 50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                    elif bar < reset_threshold:
                        # Eyebrows back to normal - allow next flap
                        self.eyebrow_raised = False
                        self.eyebrow_was_raised = False

                    # Decrease cooldown
                    if self.raised_cooldown > 0:
                        self.raised_cooldown -= 1

                    # Display BAR value with visual feedback
                    color = (0, 255, 0) if self.eyebrow_raised else (255, 255, 255)
                    cv2.putText(frame, f"BAR: {bar:.3f} / {self.raise_threshold:.3f}",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                    # Show progress bar
                    progress = min(1.0, (bar - self.baseline_bar) / (self.raise_threshold - self.baseline_bar))
                    bar_width = int(200 * max(0, progress))
                    cv2.rectangle(frame, (10, 50), (210, 70), (100, 100, 100), -1)
                    bar_color = (0, 255, 0) if bar > self.raise_threshold else (0, 165, 255)
                    cv2.rectangle(frame, (10, 50), (10 + bar_width, 70), bar_color, -1)
                    cv2.rectangle(frame, (10, 50), (210, 70), (255, 255, 255), 1)
        else:
            cv2.putText(frame, "No face detected - look at camera!", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Show camera feed
        cv2.imshow('Eyebrow Control - Raise to Flap!', frame)
        cv2.waitKey(1)

        return self.should_flap()

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
            print("\n>>> RAISE YOUR EYEBROWS to make the bird FLAP! <<<\n")

    def should_flap(self):
        if self.flap_triggered:
            self.flap_triggered = False
            return True
        return False

    def stop(self):
        self.cap.release()
        cv2.destroyAllWindows()

#------------------------------------------------------------------------------
# Modified Flappy Bird Game
#------------------------------------------------------------------------------

class FlappyEyebrow:
    def __init__(self, face_detector):
        self.face_detector = face_detector

        pygame.init()
        pygame.display.set_caption("Flappy Bird - Eyebrow Control")
        window = Window(288, 512)
        screen = pygame.display.set_mode((window.width, window.height))
        images = Images()

        self.config = GameConfig(
            screen=screen,
            clock=pygame.time.Clock(),
            fps=30,
            window=window,
            images=images,
            sounds=Sounds(),
        )

    async def start(self):
        while True:
            self.background = Background(self.config)
            self.floor = Floor(self.config)
            self.player = Player(self.config)
            self.welcome_message = WelcomeMessage(self.config)
            self.game_over_message = GameOver(self.config)
            self.pipes = Pipes(self.config)
            self.score = Score(self.config)
            await self.splash()
            await self.play()
            await self.game_over()

    async def splash(self):
        """Shows welcome splash screen - raise eyebrows to start!"""
        self.player.set_mode(PlayerMode.SHM)

        while True:
            # Process face detection
            eyebrow_flap = self.face_detector.process_frame()

            for event in pygame.event.get():
                self.check_quit_event(event)
                if self.is_tap_event(event):
                    return

            # Check for eyebrow raise to start
            if self.face_detector.calibrated and eyebrow_flap:
                return

            self.background.tick()
            self.floor.tick()
            self.player.tick()
            self.welcome_message.tick()

            # Draw instruction text
            font = pygame.font.Font(None, 24)
            if not self.face_detector.calibrated:
                text = font.render("Calibrating... Keep neutral face", True, (255, 255, 255))
            else:
                text = font.render("Raise eyebrows to START!", True, (255, 255, 0))
            text_rect = text.get_rect(center=(144, 450))
            self.config.screen.blit(text, text_rect)

            pygame.display.update()
            await asyncio.sleep(0)
            self.config.tick()

    def check_quit_event(self, event):
        if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
            self.face_detector.stop()
            pygame.quit()
            sys.exit()

    def is_tap_event(self, event):
        m_left, _, _ = pygame.mouse.get_pressed()
        space_or_up = event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP)
        screen_tap = event.type == pygame.FINGERDOWN
        return m_left or space_or_up or screen_tap

    async def play(self):
        self.score.reset()
        self.player.set_mode(PlayerMode.NORMAL)

        while True:
            # Process face detection
            eyebrow_flap = self.face_detector.process_frame()

            if self.player.collided(self.pipes, self.floor):
                return

            for i, pipe in enumerate(self.pipes.upper):
                if self.player.crossed(pipe):
                    self.score.add()

            for event in pygame.event.get():
                self.check_quit_event(event)
                if self.is_tap_event(event):
                    self.player.flap()

            # EYEBROW CONTROL: Flap when eyebrows are raised!
            if eyebrow_flap:
                self.player.flap()

            self.background.tick()
            self.floor.tick()
            self.pipes.tick()
            self.score.tick()
            self.player.tick()

            pygame.display.update()
            await asyncio.sleep(0)
            self.config.tick()

    async def game_over(self):
        """crashes the player down and shows gameover image"""
        self.player.set_mode(PlayerMode.CRASH)
        self.pipes.stop()
        self.floor.stop()

        while True:
            # Process face detection
            eyebrow_flap = self.face_detector.process_frame()

            for event in pygame.event.get():
                self.check_quit_event(event)
                if self.is_tap_event(event):
                    if self.player.y + self.player.h >= self.floor.y - 1:
                        return

            # Check for eyebrow raise to restart
            if eyebrow_flap:
                if self.player.y + self.player.h >= self.floor.y - 1:
                    return

            self.background.tick()
            self.floor.tick()
            self.pipes.tick()
            self.score.tick()
            self.player.tick()
            self.game_over_message.tick()

            # Draw restart instruction
            font = pygame.font.Font(None, 24)
            text = font.render("Raise eyebrows to RESTART!", True, (255, 255, 0))
            text_rect = text.get_rect(center=(144, 480))
            self.config.screen.blit(text, text_rect)

            self.config.tick()
            pygame.display.update()
            await asyncio.sleep(0)

#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------

async def main():
    print("=" * 50)
    print("  FLAPPY BIRD - EYEBROW CONTROL")
    print("=" * 50)
    print("\nINSTRUCTIONS:")
    print("1. Keep a NEUTRAL face during calibration (~3 seconds)")
    print("2. RAISE YOUR EYEBROWS to make the bird flap!")
    print("3. You can also use SPACEBAR or click as backup")
    print("4. Press ESC or 'q' in camera window to quit")
    print("\n" + "=" * 50 + "\n")

    # Create face detector
    face_detector = FaceDetector()

    # Start game
    game = FlappyEyebrow(face_detector)

    try:
        await game.start()
    except KeyboardInterrupt:
        pass
    finally:
        face_detector.stop()

if __name__ == "__main__":
    asyncio.run(main())
