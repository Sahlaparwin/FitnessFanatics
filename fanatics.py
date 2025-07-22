import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Webcam
cap = cv2.VideoCapture(0)

# Cooldown system
cooldown = 1.0
last_action = {"left": 0, "right": 0, "up": 0, "down": 0, "space": 0}

# Gesture thresholds
MOVE_THRESHOLD = 0.15
JUMP_HIP_THRESHOLD = 0.04
CROUCH_HIP_THRESHOLD = 0.05
HANDS_JOINED_THRESHOLD = 0.05

# Baseline setup
baseline_hip_y = None
baseline_time = None
baseline_samples = []

# Frame dimensions
frame_width = None
frame_height = None
fixed_baseline_y = 0.5  # 50% height for reference line

# Window setup
cv2.namedWindow("Fitness Fanatics - Final Version", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Fitness Fanatics - Final Version", 1280, 720)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    if frame_width is None:
        frame_height, frame_width = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)
    now = time.time()

    action_text = ""
    posture_text = ""
    calibration_text = ""
    color = (255, 255, 255)

    center_x_px = int(0.5 * frame_width)
    fixed_base_y_px = int(fixed_baseline_y * frame_height)

    if result.pose_landmarks:
        lm = result.pose_landmarks.landmark

        # Landmarks
        l_shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
        r_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        l_wrist = lm[mp_pose.PoseLandmark.LEFT_WRIST]
        r_wrist = lm[mp_pose.PoseLandmark.RIGHT_WRIST]
        l_hip = lm[mp_pose.PoseLandmark.LEFT_HIP]
        r_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP]

        # Averages
        avg_sh_x = (l_shoulder.x + r_shoulder.x) / 2
        avg_hip_y = (l_hip.y + r_hip.y) / 2
        avg_hip_y_px = int(avg_hip_y * frame_height)

        # Hands joined?
        hands_joined = (
            abs(l_wrist.x - r_wrist.x) < HANDS_JOINED_THRESHOLD and
            abs(l_wrist.y - r_wrist.y) < HANDS_JOINED_THRESHOLD
        )

        if hands_joined:
            posture_text += "Hands Joined | "
            if baseline_time is None:
                baseline_time = now
                baseline_samples = []

            if now - baseline_time < 1.0:
                baseline_samples.append(avg_hip_y)
                calibration_text = "Calibrating posture... Hold hands joined"
            elif baseline_hip_y is None:
                baseline_hip_y = min(baseline_samples)
                pyautogui.press("space")
                last_action["space"] = now
                action_text = "START GAME (SPACE)"
                color = (0, 255, 0)
        else:
            posture_text += "Hands Apart | "
            baseline_time = None
            baseline_samples = []

        # Posture Detection
        if baseline_hip_y is not None:
            delta = baseline_hip_y - avg_hip_y
            if delta > JUMP_HIP_THRESHOLD:
                posture_text += "Jumping | "
                if now - last_action["up"] > cooldown:
                    pyautogui.press("up")
                    last_action["up"] = now
                    action_text = "JUMP (UP)"
                    color = (0, 255, 255)
            elif -delta > CROUCH_HIP_THRESHOLD:
                posture_text += "Crouching | "
                if now - last_action["down"] > cooldown:
                    pyautogui.press("down")
                    last_action["down"] = now
                    action_text = "CROUCH (DOWN)"
                    color = (0, 128, 255)
            else:
                posture_text += "Standing | "

        # Movement: Left / Right
        if avg_sh_x < 0.4:
            posture_text += "Left"
            if now - last_action["left"] > cooldown:
                pyautogui.press("left")
                last_action["left"] = now
                action_text = "MOVE LEFT"
                color = (255, 255, 0)
        elif avg_sh_x > 0.6:
            posture_text += "Right"
            if now - last_action["right"] > cooldown:
                pyautogui.press("right")
                last_action["right"] = now
                action_text = "MOVE RIGHT"
                color = (0, 255, 0)
        else:
            posture_text += "Center"

        # Draw pose landmarks (white points, red connections)
        mp_drawing.draw_landmarks(
            frame,
            result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1)
        )

        # Draw current hip position (grey dashed line, thinner)
        for x in range(0, frame_width, 20):
            cv2.line(frame, (x, avg_hip_y_px), (x + 10, avg_hip_y_px), (180, 180, 180), 1)

    # Draw vertical center line (red, thinner)
    cv2.line(frame, (center_x_px, 0), (center_x_px, frame_height), (0, 0, 255), 1)

    # Fixed reference horizontal line (red, thinner)
    cv2.line(frame, (0, fixed_base_y_px), (frame_width, fixed_base_y_px), (0, 0, 255), 1)

    # Show texts
    if posture_text:
        cv2.putText(frame, f"Posture: {posture_text}", (30, frame_height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    if action_text:
        cv2.putText(frame, action_text, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    if calibration_text:
        cv2.putText(frame, calibration_text, (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 255), 2)

    # Show window
    cv2.imshow("Fitness Fanatics - Final Version", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




