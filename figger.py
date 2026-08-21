import os
import time

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # 壓制 oneDNN 警告訊息

import cv2
import mediapipe as mp

# =========================
# MediaPipe 初始化 (0.10.x 標準載入方式)
# =========================
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# =========================
# 參數與攝影機初始化
# =========================
CAMERA_ID = 0
WINDOW_NAME = "oxxostudio"
FRAME_W, FRAME_H = 1280, 720
MAX_NUM_HANDS = 2

cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

# 固定視窗位置，防止跑到螢幕外
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.moveWindow(WINDOW_NAME, 200, 100)

print("手部偵測啟動中... (按下 'q' 或按 Ctrl+C 可退出)")

# =========================
# 主迴圈
# =========================
try:
    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        prev_time = time.time()

        while True:
            ret, img = cap.read()
            if not ret:
                print("Cannot receive frame")
                break

            # 鏡像翻轉
            img = cv2.flip(img, 1)

            # MediaPipe 辨識 (需要 RGB 格式)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_rgb.flags.writeable = False
            results = hands.process(img_rgb)
            img_rgb.flags.writeable = True

            # 繪製手部骨架
            hand_count = 0
            if results.multi_hand_landmarks:
                hand_count = len(results.multi_hand_landmarks)
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        img,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )

            # 計算 FPS
            curr_time = time.time()
            time_diff = curr_time - prev_time
            fps = 1 / time_diff if time_diff > 0 else 0.0
            prev_time = curr_time

            # 顯示資訊文字
            cv2.putText(
                img,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                img,
                f"Hands: {hand_count}",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

            cv2.imshow(WINDOW_NAME, img)

            # 按鍵監聽
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("收到關閉指令 'q'")
                break

except KeyboardInterrupt:
    print("\n[系統提示] 偵測到 Ctrl+C，程式已安全關閉。")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("資源已關閉完成。")