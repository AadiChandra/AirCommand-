import cv2
import mediapipe as mp
import math
import numpy as np
import time
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import screen_brightness_control as sbc
import ctypes

# --------------------- Multimedia key helpers (Windows) ---------------------
# Uses SendInput to send multimedia keys (works without extra packages)
# Virtual-Key codes
VK_MEDIA_PLAY_PAUSE = 0xB3
VK_MEDIA_NEXT_TRACK = 0xB0
VK_MEDIA_PREV_TRACK = 0xB1

# Send a single key event using keybd_event (compatible with older Windows)
user32 = ctypes.WinDLL('user32', use_last_error=True)

def send_vk(vk_code):
    # key down
    user32.keybd_event(vk_code, 0, 0, 0)
    time.sleep(0.05)
    # key up
    user32.keybd_event(vk_code, 0, 2, 0)

# --------------------- Pycaw audio setup ---------------------
devices = AudioUtilities.GetDeviceEnumerator()
default_device = devices.GetDefaultAudioEndpoint(0, 1)
interface = default_device.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()
minVol, maxVol = volRange[0], volRange[1]

# --------------------- Mediapipe setup ---------------------
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Utility: count fingers up using landmark geometry
FINGER_TIP_IDS = [4, 8, 12, 16, 20]

def fingers_up(lmList):
    # Returns a list of 5 ints (1 if finger up else 0) for [thumb,index,middle,ring,pinky]
    fingers = []
    if not lmList:
        return [0,0,0,0,0]
    # Thumb: compare tip and ip landmarks in x for handedness-agnostic approach use x
    # But since mediapipe doesn't return handedness here, we use relative x
    # Thumb: tip (4) x < ip (3) x -> likely open for right hand in camera mirror. We'll use y-axis-based check for robustness with simple heuristic
    # We'll check if thumb tip is away from palm center in x
    # For simplicity: compare tip to mcp (2)
    if lmList[4][1] < lmList[3][1]:
        fingers.append(1)
    else:
        fingers.append(0)
    # Other fingers: tip y < pip y => finger up
    for id in [8, 12, 16, 20]:
        tip_y = lmList[id][2]
        pip_y = lmList[id - 2][2]
        fingers.append(1 if tip_y < pip_y else 0)
    return fingers

# Utility: detect fist (all fingers down)
def is_fist(fingers):
    return sum(fingers) == 0

# Utility: detect peace sign (index & middle up, others down)
def is_peace(fingers):
    return fingers[1] == 1 and fingers[2] == 1 and fingers[0] == 0 and fingers[3] == 0 and fingers[4] == 0

# --------------------- Main loop ---------------------
wCam, hCam = 1280, 720
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# Gesture debounce timers
last_mute_toggle = 0
mute_cooldown = 1.0  # seconds
last_play_toggle = 0
play_cooldown = 1.0
last_brightness_update = 0
brightness_cooldown = 0.05

# State
is_muted = False

with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.6, min_tracking_confidence=0.6, max_num_hands=1) as hands:
    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        lmList = []
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            h, w, c = img.shape
            for id, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

        # Default UI elements
        overlay = img.copy()

        if lmList:
            fingers = fingers_up({i: None for i in range(0)}) if False else fingers_up(lmList)
            # Volume control: thumb (4) and index (8)
            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]
            cv2.circle(img, (x1, y1), 12, (255, 255, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 12, (255, 255, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            length = math.hypot(x2 - x1, y2 - y1)

            # Map length to volume range
            vol = np.interp(length, [30, 220], [minVol, maxVol])
            volBar = np.interp(length, [30, 220], [400, 150])
            volPer = np.interp(length, [30, 220], [0, 100])
            # Set volume only if not muted
            if not is_muted:
                volume.SetMasterVolumeLevel(vol, None)

            # Brightness control: thumb (4) and pinky (20)
            bx1, by1 = lmList[4][1], lmList[4][2]
            bx2, by2 = lmList[20][1], lmList[20][2]
            blength = math.hypot(bx2 - bx1, by2 - by1)
            briPer = np.interp(blength, [30, 260], [0, 100])

            # Mute gesture: fist
            if is_fist(fingers) and (time.time() - last_mute_toggle) > mute_cooldown:
                # toggle mute
                try:
                    current_mute = ctypes.c_uint()
                    # Some systems support GetMute; using try/except
                    try:
                        is_muted_state = volume.GetMute()
                    except Exception:
                        is_muted_state = is_muted
                    if is_muted_state:
                        volume.SetMute(0, None)
                        is_muted = False
                    else:
                        volume.SetMute(1, None)
                        is_muted = True
                    last_mute_toggle = time.time()
                except Exception as e:
                    print('Mute toggle failed:', e)

            # Play/Pause gesture: peace sign
            if is_peace(fingers) and (time.time() - last_play_toggle) > play_cooldown:
                send_vk(VK_MEDIA_PLAY_PAUSE)
                last_play_toggle = time.time()

            # Update brightness when index finger up alone (or when pinky-thumb gesture used)
            # We'll update brightness when pinky and thumb are moving (blength) and cooldown passed
            if (time.time() - last_brightness_update) > brightness_cooldown:
                # Only update brightness when pinky or ring visible to avoid accidental changes
                if fingers[4] == 1 or True:
                    try:
                        sbc.set_brightness(int(briPer))
                    except Exception:
                        pass
                last_brightness_update = time.time()

            # Stylish volume bar (gradient-like using rectangles)
            # Background rounded rectangle
            cv2.rectangle(overlay, (40, 120), (110, 420), (50, 50, 50), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
            # Filled volume level
            cv2.rectangle(img, (50, int(np.interp(volPer, [0,100],[400,150]))), (100, 400), (0, 255, 0), -1)
            # Border
            cv2.rectangle(img, (40, 120), (110, 420), (255, 255, 255), 2)
            cv2.putText(img, f'Vol: {int(volPer)}%', (45, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(img, f'Bri: {int(briPer)}%', (130, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(img, f'Muted: {is_muted}', (300, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255) if is_muted else (0,255,0), 2)

        # Show instructions
        cv2.rectangle(img, (10,10), (540,90), (0,0,0), -1)
        cv2.putText(img, 'Controls: Thumb+Index -> Volume | Thumb+Pinky -> Brightness', (20,35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        cv2.putText(img, 'Fist -> Mute toggle | Peace sign (index+middle) -> Play/Pause', (20,65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        cv2.imshow('Gesture Controller', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
