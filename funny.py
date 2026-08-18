import time
import cv2
import numpy as np

# =========================
# 常數與參數設定
# =========================
CAMERA_ID = 0
WINDOW_NAME = "oxxostudio"
FRAME_W = 640
FRAME_H = 360
CROP_SIZE = FRAME_H  # 正方形邊長 (360)
SCAN_STEP = 2  # 每次移動像素數
MOTION_THRESHOLD = 1000000  # 晃動觸發門檻
THRESH_BINARY_VALUE = 25
COOLDOWN_SEC = 2.0  # 掃描後的冷卻時間 (秒)

# =========================
# 攝影機與畫布初始化
# =========================
cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.moveWindow(WINDOW_NAME, 200, 100)

output = np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)
a = 1
run = False
x = 0
prev_gray = None
last_capture_time = 0

print("程式啟動中... (按下 'q' 或在 Terminal 按 Ctrl+C 可退出程式)")

# =========================
# 主迴圈
# =========================
try:
    while True:
        ret, img = cap.read()
        if not ret:
            print("Cannot receive frame")
            break

        # --- 前處理 ---
        img = cv2.resize(img, (FRAME_W, FRAME_H))
        img = cv2.flip(img, 1)

        # 取中間正方形區域
        start_x = (FRAME_W - FRAME_H) // 2
        end_x = start_x + FRAME_H
        img = img[:, start_x:end_x]

        # 灰階 + 模糊
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        current_time = time.time()
        in_cooldown = (current_time - last_capture_time) < COOLDOWN_SEC

        # --- 1. 晃動偵測 ---
        if not run:
            if in_cooldown:
                cv2.putText(
                    img,
                    "Cooldown...",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )
            else:
                cv2.putText(
                    img,
                    "Wave to start!",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

                if prev_gray is not None:
                    frame_delta = cv2.absdiff(prev_gray, gray)
                    _, thresh = cv2.threshold(
                        frame_delta, THRESH_BINARY_VALUE, 255, cv2.THRESH_BINARY
                    )
                    motion_score = cv2.countNonZero(thresh)

                    # 晃動分數達標且非冷卻狀態才啟動
                    if motion_score * 255 > MOTION_THRESHOLD:
                        x = 0
                        run = True
                        output[:] = 0  # 清空畫布
                        print("偵測到晃動，開始掃描！")

            prev_gray = gray.copy()

        # --- 2. 按鍵監聽 ---
        keyCode = cv2.waitKey(1) & 0xFF
        if keyCode == ord("q"):
            print("收到關閉指令 'q'")
            break
        elif keyCode == ord("a") and not run and not in_cooldown:
            x = 0
            run = True
            output[:] = 0
            print("手動啟動掃描")

        # --- 3. 掃描與合成 ---
        if run:
            x_end = min(x + SCAN_STEP, CROP_SIZE)

            # 將目前掃描條寫入 output
            output[:, x:x_end] = img[:, x:x_end]

            # 繪製紅線
            line_x = min(x + 5, CROP_SIZE - 1)
            cv2.line(
                img, (line_x, 0), (line_x, CROP_SIZE), (0, 0, 255), 5
            )

            # 將已掃描部分合成回畫面
            img[:, :x_end] = output[:, :x_end]
            x = x_end

            # 掃描完成處置
            if x >= CROP_SIZE:
                filename = f"oxxo-{a}.jpg"
                cv2.imwrite(filename, img)
                print(f"已自動儲存 {filename}")
                a += 1
                run = False
                prev_gray = None
                last_capture_time = time.time()

        # --- 4. 顯示畫面 ---
        cv2.imshow(WINDOW_NAME, img)

except KeyboardInterrupt:
    print("\n[系統提示] 偵測到使用者在 Terminal 按下 Ctrl+C，程式已安全結束。")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("資源已關閉完成。")
