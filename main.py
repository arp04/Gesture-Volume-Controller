import cv2
import mediapipe as mp
import numpy as np
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
import time


# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


# Initialize MediaPipe parameters
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Initialize PyCaw for volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)

# Function to detect the circular motion direction of the index finger
def get_circle_direction(points):
    center = np.mean(points, axis=0)
    vectors = [point - center for point in points]
    angles = np.arctan2([vector[1] for vector in vectors], [vector[0] for vector in vectors])
    angles = np.unwrap(angles)
    direction = np.sign(np.mean(np.diff(angles)))
    # Reverse the direction
    return 'clockwise' if direction > 0 else 'counterclockwise'

# Function to control volume
def control_system(action):
    current_volume = volume.GetMasterVolumeLevelScalar()
    if action == 'volume_up':
        volume.SetMasterVolumeLevelScalar(min(current_volume + 0.02, 1.0), None)
    elif action == 'volume_down':
        volume.SetMasterVolumeLevelScalar(max(current_volume - 0.02, 0.0), None)

# Start capturing video from the camera
cap = cv2.VideoCapture(0)

points_buffer = []
buffer_size = 20  # Number of points for circle detection
last_action_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            index_tip = hand_landmarks.landmark[8]
            points_buffer.append((index_tip.x, index_tip.y))

            if len(points_buffer) > buffer_size:
                points_buffer.pop(0)

            if len(points_buffer) == buffer_size:
                current_time = time.time()
                if current_time - last_action_time > 0.05:  # Increase volume change speed
                    direction = get_circle_direction(points_buffer)
                    if direction == 'clockwise':
                        control_system('volume_down')
                    elif direction == 'counterclockwise':
                        control_system('volume_up')
                    last_action_time = current_time

    cv2.imshow('Hand Gesture Control', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
