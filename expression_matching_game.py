import cv2
import numpy as np
import time
import random

# Install pygame for sound effects: pip install pygame
try:
    from pygame import mixer
except ModuleNotFoundError:
    mixer = None

#------------------------------------------------------------------------------
# 1. Initializations.
#------------------------------------------------------------------------------

# Model file paths.
MODEL_PATH = './model/res10_300x300_ssd_iter_140000.caffemodel'
CONFIG_PATH = './model/deploy.prototxt'
LBF_MODEL = './model/lbfmodel.yaml'

# Create a face detector network instance.
net = cv2.dnn.readNetFromCaffe(CONFIG_PATH, MODEL_PATH)

# Create the landmark detector instance.
landmarkDetector = cv2.face.createFacemarkLBF()
landmarkDetector.loadModel(LBF_MODEL)

# Initialize video capture object.
cap = cv2.VideoCapture(0)

# Game state variables
GAME_STATE = 'CALIBRATION'  # CALIBRATION, READY, PLAYING, FEEDBACK, GAMEOVER
current_emotion = None
target_emotion = None
round_number = 0
total_score = 0
round_start_time = 0
calibration_frames = 0
match_threshold = 75  # Percentage match needed to succeed

# Emotion definitions with feature ranges (will be calibrated)
EMOTIONS = {
    'NEUTRAL': {
        'name': 'Neutral',
        'emoji': '😐',
        'description': 'Keep a relaxed, neutral face'
    },
    'HAPPY': {
        'name': 'Happy',
        'emoji': '😊',
        'description': 'Smile wide!'
    },
    'SURPRISED': {
        'name': 'Surprised',
        'emoji': '😮',
        'description': 'Open your eyes and mouth wide!'
    },
    'WINK': {
        'name': 'Wink',
        'emoji': '😉',
        'description': 'Close one eye and smile'
    }
}

# Calibration baselines (neutral face)
baseline_ear = 0
baseline_bar = 0
baseline_mar = 0
baseline_smile = 0

#------------------------------------------------------------------------------
# 2. Function definitions.
#------------------------------------------------------------------------------

# Face detection function
def detect_faces(image, detection_threshold=0.70):
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123])
    net.setInput(blob)
    detections = net.forward()
    faces = []
    img_h = image.shape[0]
    img_w = image.shape[1]

    for detection in detections[0][0]:
        if detection[2] >= detection_threshold:
            left   = detection[3] * img_w
            top    = detection[4] * img_h
            right  = detection[5] * img_w
            bottom = detection[6] * img_h
            face_w = right - left
            face_h = bottom - top
            face_roi = (left, top, face_w, face_h)
            faces.append(face_roi)

    return np.array(faces).astype(int)

def get_primary_face(faces, frame_h, frame_w):
    primary_face_index = None
    face_height_max = 0
    for idx in range(len(faces)):
        face = faces[idx]
        x1 = face[0]
        y1 = face[1]
        x2 = x1 + face[2]
        y2 = y1 + face[3]
        if x1 > frame_w or y1 > frame_h or x2 > frame_w or y2 > frame_h:
            continue
        if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
            continue
        if face[3] > face_height_max:
            primary_face_index = idx
            face_height_max = face[3]

    if primary_face_index is not None:
        primary_face = faces[primary_face_index]
    else:
        primary_face = None

    return primary_face

def calculate_distance(A, B):
    distance = ((A[0] - B[0])**2+(A[1] - B[1])**2)**0.5
    return distance

# Eye Aspect Ratio (for blinks and eye openness)
def get_eye_aspect_ratio(landmarks):
    vert_dist_1right = calculate_distance(landmarks[37], landmarks[41])
    vert_dist_2right = calculate_distance(landmarks[38], landmarks[40])
    vert_dist_1left  = calculate_distance(landmarks[43], landmarks[47])
    vert_dist_2left  = calculate_distance(landmarks[44], landmarks[46])

    horz_dist_right  = calculate_distance(landmarks[36], landmarks[39])
    horz_dist_left = calculate_distance(landmarks[42], landmarks[45])

    EAR_left = (vert_dist_1left + vert_dist_2left) / (2.0 * horz_dist_left)
    EAR_right = (vert_dist_1right + vert_dist_2right) / (2.0 * horz_dist_right)

    ear = (EAR_left + EAR_right) / 2
    return ear, EAR_left, EAR_right

# Eyebrow Aspect Ratio (for eyebrow raises)
def get_eyebrow_aspect_ratio(landmarks):
    right_eyebrow_height = calculate_distance(landmarks[19], landmarks[37])
    left_eyebrow_height = calculate_distance(landmarks[24], landmarks[44])

    right_eyebrow_width = calculate_distance(landmarks[17], landmarks[21])
    left_eyebrow_width = calculate_distance(landmarks[22], landmarks[26])

    right_bar = right_eyebrow_height / right_eyebrow_width
    left_bar = left_eyebrow_height / left_eyebrow_width

    bar = (right_bar + left_bar) / 2
    return bar

# Mouth Aspect Ratio (for open mouth / surprise)
def get_mouth_aspect_ratio(landmarks):
    # Vertical: top lip to bottom lip (multiple points for accuracy)
    vert_dist_1 = calculate_distance(landmarks[51], landmarks[57])  # Center
    vert_dist_2 = calculate_distance(landmarks[62], landmarks[66])  # Inner center

    # Horizontal: left corner to right corner
    horz_dist = calculate_distance(landmarks[48], landmarks[54])

    # MAR is ratio of vertical to horizontal
    mar = (vert_dist_1 + vert_dist_2) / (2.0 * horz_dist)
    return mar

# Smile Detection (based on mouth corner positions)
def get_smile_score(landmarks):
    # Measure if mouth corners are raised
    # Compare mouth corner height to mouth center height

    # Left corner (48), Right corner (54), Top center (51), Bottom center (57)
    left_corner = landmarks[48]
    right_corner = landmarks[54]
    mouth_center_y = (landmarks[51][1] + landmarks[57][1]) / 2

    # Calculate how much corners are raised relative to center
    left_raise = mouth_center_y - left_corner[1]
    right_raise = mouth_center_y - right_corner[1]

    # Also measure mouth width (smiles are wider)
    mouth_width = calculate_distance(landmarks[48], landmarks[54])

    # Normalize by mouth width
    smile_score = (left_raise + right_raise) / (2.0 * mouth_width)

    return smile_score

# Classify emotion based on facial features
def classify_emotion(ear, ear_left, ear_right, bar, mar, smile, baseline_ear, baseline_bar, baseline_mar, baseline_smile):
    features = {
        'ear': ear,
        'ear_left': ear_left,
        'ear_right': ear_right,
        'bar': bar,
        'mar': mar,
        'smile': smile
    }

    # Calculate relative changes from baseline
    ear_ratio = ear / baseline_ear if baseline_ear > 0 else 1
    bar_ratio = bar / baseline_bar if baseline_bar > 0 else 1
    mar_ratio = mar / baseline_mar if baseline_mar > 0 else 1
    smile_ratio = smile / baseline_smile if baseline_smile > 0.001 else 1

    # Determine current emotion with more forgiving thresholds

    # WINK: Asymmetric eyes (check this first as it's most distinct)
    if abs(ear_left - ear_right) > 0.08:
        # One eye should be significantly more closed
        if min(ear_left, ear_right) < baseline_ear * 0.6:
            return 'WINK', features

    # SURPRISED: Wide eyes, raised eyebrows, open mouth
    # More forgiving - at least 2 of 3 features should be elevated
    surprised_score = 0
    if ear_ratio > 1.10:  # Eyes wider than normal
        surprised_score += 1
    if bar_ratio > 1.10:  # Eyebrows raised
        surprised_score += 1
    if mar_ratio > 1.2:   # Mouth open
        surprised_score += 1

    if surprised_score >= 2:
        return 'SURPRISED', features

    # HAPPY: Smile detection (more lenient)
    # Check both smile score AND mouth width increase
    if smile_ratio > 1.15 or mar_ratio > 1.1:
        # Additional check: smile should have positive value
        if smile > 0.05:
            return 'HAPPY', features

    # NEUTRAL: Everything close to baseline (within 15% tolerance)
    neutral_score = 0
    if 0.85 <= ear_ratio <= 1.15:
        neutral_score += 1
    if 0.85 <= bar_ratio <= 1.15:
        neutral_score += 1
    if 0.85 <= mar_ratio <= 1.15:
        neutral_score += 1

    if neutral_score >= 2:
        return 'NEUTRAL', features

    # Default to neutral if uncertain
    return 'NEUTRAL', features

# Calculate match score between current and target emotion
def calculate_match_score(current_emotion, target_emotion):
    if current_emotion == target_emotion:
        return 100

    # Partial credit for related emotions
    if current_emotion == 'HAPPY' and target_emotion == 'WINK':
        return 50
    if current_emotion == 'WINK' and target_emotion == 'HAPPY':
        return 50
    if current_emotion == 'NEUTRAL' and target_emotion in ['HAPPY', 'SURPRISED', 'WINK']:
        return 30

    return 0

# Draw UI elements
def draw_ui(frame, game_state, target_emotion, current_emotion, match_score, total_score, round_number, time_remaining):
    h, w = frame.shape[:2]

    # Create semi-transparent overlay for UI
    overlay = frame.copy()

    if game_state == 'CALIBRATION':
        # Calibration instructions
        cv2.rectangle(overlay, (50, 50), (w-50, 200), (0, 0, 0), -1)
        frame_alpha = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        cv2.putText(frame_alpha, "CALIBRATION", (w//2 - 150, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.putText(frame_alpha, "Keep a neutral face...", (w//2 - 200, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
        return frame_alpha

    elif game_state == 'READY':
        # Show next target emotion
        cv2.rectangle(overlay, (50, h//2 - 150), (w-50, h//2 + 150), (0, 0, 0), -1)
        frame_alpha = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)

        if target_emotion:
            emoji = EMOTIONS[target_emotion]['emoji']
            description = EMOTIONS[target_emotion]['description']

            cv2.putText(frame_alpha, f"GET READY!", (w//2 - 150, h//2 - 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)
            cv2.putText(frame_alpha, f"{emoji} {description}", (w//2 - 200, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            cv2.putText(frame_alpha, "Starting in 3...", (w//2 - 100, h//2 + 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
        return frame_alpha

    elif game_state == 'PLAYING':
        # Show target emotion and progress
        if target_emotion:
            emoji = EMOTIONS[target_emotion]['emoji']
            description = EMOTIONS[target_emotion]['description']

            # Top banner with target
            cv2.rectangle(overlay, (0, 0), (w, 150), (0, 0, 0), -1)
            frame_alpha = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

            cv2.putText(frame_alpha, f"Make this face: {emoji} {EMOTIONS[target_emotion]['name']}",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            cv2.putText(frame_alpha, f"Time: {int(time_remaining)}s",
                        (w - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            # Current detection
            if current_emotion:
                current_emoji = EMOTIONS[current_emotion]['emoji']
                cv2.putText(frame_alpha, f"You: {current_emoji} {EMOTIONS[current_emotion]['name']}",
                            (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)

            # Match progress bar
            bar_width = int((w - 100) * match_score / 100)
            bar_color = (0, 255, 0) if match_score >= match_threshold else (0, 165, 255)
            cv2.rectangle(frame_alpha, (50, 120), (w - 50, 140), (100, 100, 100), -1)
            cv2.rectangle(frame_alpha, (50, 120), (50 + bar_width, 140), bar_color, -1)
            cv2.putText(frame_alpha, f"{int(match_score)}%", (w//2 - 30, 135),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Score
            cv2.putText(frame_alpha, f"Score: {total_score}",
                        (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame_alpha, f"Round: {round_number}",
                        (w - 200, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            return frame_alpha

    elif game_state == 'FEEDBACK':
        # Show round result
        cv2.rectangle(overlay, (w//4, h//2 - 100), (3*w//4, h//2 + 100), (0, 0, 0), -1)
        frame_alpha = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)

        if match_score >= match_threshold:
            cv2.putText(frame_alpha, "GREAT JOB!", (w//2 - 150, h//2 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.putText(frame_alpha, f"Match: {int(match_score)}%", (w//2 - 100, h//2 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            cv2.putText(frame_alpha, "TRY AGAIN!", (w//2 - 130, h//2 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 100, 255), 3)
            cv2.putText(frame_alpha, f"Match: {int(match_score)}%", (w//2 - 100, h//2 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return frame_alpha

    return frame

# Visualize facial landmarks
def visualize_landmarks(frame, landmarks):
    # Eyes (green)
    for i in range(36, 48):
        cv2.circle(frame, tuple(landmarks[i].astype('int')), 2, (0, 255, 0), -1)

    # Eyebrows (blue)
    for i in range(17, 27):
        cv2.circle(frame, tuple(landmarks[i].astype('int')), 2, (255, 0, 0), -1)

    # Mouth (red)
    for i in range(48, 68):
        cv2.circle(frame, tuple(landmarks[i].astype('int')), 2, (0, 0, 255), -1)

#------------------------------------------------------------------------------
# 3. Game Logic.
#------------------------------------------------------------------------------

if __name__ == "__main__":
    ret, frame = cap.read()
    frame_h = frame.shape[0]
    frame_w = frame.shape[1]

    # Calibration variables
    calib_frames_needed = 30
    sum_ear = 0
    sum_bar = 0
    sum_mar = 0
    sum_smile = 0

    # Game timing
    round_duration = 8  # seconds per round
    feedback_duration = 2  # seconds to show feedback
    ready_duration = 3  # seconds to prepare
    total_rounds = 5

    # Success tracking - need to maintain expression for this long
    success_hold_time = 3  # seconds
    success_start_time = None
    success_achieved = False

    state_start_time = time.time()
    match_score = 0
    current_emotion = None

    print("=== FACIAL EXPRESSION MATCHING GAME ===")
    print("Match your facial expression to the target!")
    print("Press 'd' to toggle debug mode, 'q' to quit\n")

    # Debug mode toggle
    debug_mode = False

    while cap.isOpened():
        ret, frame = cap.read()

        if ret != True:
            break

        # Detect face
        faces = detect_faces(frame, detection_threshold=.90)

        if len(faces) > 0:
            primary_face = get_primary_face(faces, frame_h, frame_w)

            if primary_face is not None:
                cv2.rectangle(frame, primary_face, (0, 255, 0), 2)

                # Detect landmarks
                retval, landmarksList = landmarkDetector.fit(frame, np.expand_dims(primary_face, 0))

                if retval:
                    landmarks = landmarksList[0][0]

                    # Visualize landmarks
                    visualize_landmarks(frame, landmarks)

                    # Calculate all facial metrics
                    ear, ear_left, ear_right = get_eye_aspect_ratio(landmarks)
                    bar = get_eyebrow_aspect_ratio(landmarks)
                    mar = get_mouth_aspect_ratio(landmarks)
                    smile = get_smile_score(landmarks)

                    # Game state machine
                    if GAME_STATE == 'CALIBRATION':
                        if calibration_frames < calib_frames_needed:
                            calibration_frames += 1
                            sum_ear += ear
                            sum_bar += bar
                            sum_mar += mar
                            sum_smile += smile
                        else:
                            # Calibration complete
                            baseline_ear = sum_ear / calib_frames_needed
                            baseline_bar = sum_bar / calib_frames_needed
                            baseline_mar = sum_mar / calib_frames_needed
                            baseline_smile = sum_smile / calib_frames_needed

                            print(f"Calibration complete!")
                            print(f"Baseline - EAR: {baseline_ear:.3f}, BAR: {baseline_bar:.3f}, MAR: {baseline_mar:.3f}, Smile: {baseline_smile:.3f}\n")

                            GAME_STATE = 'READY'
                            round_number = 1
                            total_score = 0
                            target_emotion = random.choice(list(EMOTIONS.keys()))
                            state_start_time = time.time()

                    elif GAME_STATE == 'READY':
                        # Wait for ready period
                        if time.time() - state_start_time > ready_duration:
                            GAME_STATE = 'PLAYING'
                            state_start_time = time.time()
                            success_achieved = False
                            success_start_time = None
                            print(f"\nRound {round_number}: Match {EMOTIONS[target_emotion]['emoji']} {EMOTIONS[target_emotion]['name']}")

                    elif GAME_STATE == 'PLAYING':
                        # Classify current emotion
                        current_emotion, features = classify_emotion(
                            ear, ear_left, ear_right, bar, mar, smile,
                            baseline_ear, baseline_bar, baseline_mar, baseline_smile
                        )

                        # Calculate match score
                        match_score = calculate_match_score(current_emotion, target_emotion)

                        # Track success (need to hold for success_hold_time seconds)
                        if match_score >= match_threshold and not success_achieved:
                            if success_start_time is None:
                                success_start_time = time.time()
                                print(f"\nGood match! Hold this expression for {success_hold_time} seconds...")
                            else:
                                hold_duration = time.time() - success_start_time
                                if hold_duration >= success_hold_time:
                                    success_achieved = True
                                    round_score = 100
                                    total_score += round_score
                                    print(f"✓ SUCCESS! +{round_score} points (Total: {total_score})")
                        elif match_score < match_threshold:
                            # Lost the match, reset timer
                            if success_start_time is not None:
                                print(f"Lost the match - try again!")
                            success_start_time = None

                        # Check time
                        time_elapsed = time.time() - state_start_time
                        time_remaining = round_duration - time_elapsed

                        if time_remaining <= 0:
                            # Round over
                            if not success_achieved:
                                print(f"Failed - Didn't hold expression long enough")

                            GAME_STATE = 'FEEDBACK'
                            state_start_time = time.time()

                    elif GAME_STATE == 'FEEDBACK':
                        # Show feedback
                        if time.time() - state_start_time > feedback_duration:
                            round_number += 1

                            if round_number > total_rounds:
                                # Game over
                                print(f"\n=== GAME OVER ===")
                                print(f"Final Score: {total_score}/{total_rounds * 100}")
                                print(f"Percentage: {int(total_score / (total_rounds * 100) * 100)}%")
                                GAME_STATE = 'GAMEOVER'
                                break
                            else:
                                # Next round
                                target_emotion = random.choice(list(EMOTIONS.keys()))
                                GAME_STATE = 'READY'
                                state_start_time = time.time()
                                match_score = 0
                                success_achieved = False
                                success_start_time = None

                    # Draw UI
                    frame = draw_ui(frame, GAME_STATE, target_emotion, current_emotion,
                                   match_score if GAME_STATE == 'PLAYING' else 0,
                                   total_score, round_number,
                                   round_duration - (time.time() - state_start_time) if GAME_STATE == 'PLAYING' else 0)

                    # Add debug info if enabled
                    if debug_mode and GAME_STATE == 'PLAYING':
                        h, w = frame.shape[:2]
                        debug_y = h - 150
                        cv2.putText(frame, f"DEBUG MODE", (w - 300, debug_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        cv2.putText(frame, f"EAR: {ear:.3f} (base: {baseline_ear:.3f})",
                                    (w - 300, debug_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.putText(frame, f"BAR: {bar:.3f} (base: {baseline_bar:.3f})",
                                    (w - 300, debug_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.putText(frame, f"MAR: {mar:.3f} (base: {baseline_mar:.3f})",
                                    (w - 300, debug_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.putText(frame, f"Smile: {smile:.3f} (base: {baseline_smile:.3f})",
                                    (w - 300, debug_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    # Show hold timer if matching
                    if GAME_STATE == 'PLAYING' and success_start_time is not None and not success_achieved:
                        hold_duration = time.time() - success_start_time
                        hold_remaining = success_hold_time - hold_duration
                        h, w = frame.shape[:2]
                        cv2.putText(frame, f"HOLD for {hold_remaining:.1f}s!", (w//2 - 150, h - 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        cv2.imshow('Expression Matching Game', frame)

        k = cv2.waitKey(1)
        if k == ord('q'):
            break
        elif k == ord('d'):
            debug_mode = not debug_mode
            print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()
