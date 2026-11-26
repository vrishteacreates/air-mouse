
import cv2
import mediapipe as mp
import pyautogui
import math

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Screen dimensions
screen_width, screen_height = pyautogui.size()

# Webcam
cap = cv2.VideoCapture(0)

# Hand detection
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Smooth cursor movement
smoothening = 5
prev_x, prev_y = 0, 0
curr_x, curr_y = 0, 0

# Click control
clicking = False

while cap.isOpened():

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    height, width, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Index finger tip
            index = hand_landmarks.landmark[8]

            # Thumb tip
            thumb = hand_landmarks.landmark[4]

            # Convert coordinates
            x = int(index.x * width)
            y = int(index.y * height)

            # Map camera coordinates to screen
            target_x = pyautogui.map_value(
                index.x, 0.1, 0.9, 0, screen_width
            ) if hasattr(pyautogui, "map_value") else int(
                (index.x - 0.1) / 0.8 * screen_width
            )

            target_y = int(
                (index.y - 0.1) / 0.8 * screen_height
            )

            target_x = max(0, min(screen_width - 1, target_x))
            target_y = max(0, min(screen_height - 1, target_y))

            # Smooth movement
            curr_x = prev_x + (target_x - prev_x) / smoothening
            curr_y = prev_y + (target_y - prev_y) / smoothening

            pyautogui.moveTo(int(curr_x), int(curr_y))

            prev_x, prev_y = curr_x, curr_y

            # Calculate distance between thumb and index
            distance = math.hypot(
                thumb.x - index.x,
                thumb.y - index.y
            )

            # Pinch detection
            if distance < 0.045:

                if not clicking:
                    pyautogui.click()
                    clicking = True

                cv2.putText(
                    frame,
                    "CLICK",
                    (x, y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

            else:
                clicking = False

    # Display camera
    cv2.imshow("AIR MOUSE", frame)

    # Exit using Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
hands.close()
cv2.destroyAllWindows()
