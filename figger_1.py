import cv2
import mediapipe as mp
import random
import time
import math

# =========================================================
# MediaPipe
# =========================================================
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


# =========================================================
# 遊戲設定
# =========================================================
CAMERA_ID = 0

FRAME_WIDTH = 960
FRAME_HEIGHT = 540

TARGET_SIZE = 70

# 手指平滑係數
# 越小越穩定，但反應越慢
SMOOTHING_ALPHA = 0.35

# 遊戲時間
GAME_TIME = 30


# =========================================================
# 攝影機
# =========================================================
cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)


# =========================================================
# 建立遊戲目標
# =========================================================
def create_target(width, height):
    margin = 80

    x = random.randint(
        margin,
        width - TARGET_SIZE - margin
    )

    y = random.randint(
        margin,
        height - TARGET_SIZE - margin
    )

    return x, y


# =========================================================
# 判斷手指是否碰到目標
# =========================================================
def is_inside_target(x, y, target_x, target_y):

    return (
        target_x <= x <= target_x + TARGET_SIZE
        and
        target_y <= y <= target_y + TARGET_SIZE
    )


# =========================================================
# 計算 FPS
# =========================================================
prev_time = time.time()


# =========================================================
# 遊戲狀態
# =========================================================
target_x = 0
target_y = 0

score = 0

target_initialized = False

smooth_x = None
smooth_y = None

game_started = False
game_start_time = None


# =========================================================
# MediaPipe Hands
# =========================================================
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=0,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
) as hands:

    while True:

        # =================================================
        # 讀取攝影機
        # =================================================
        ret, frame = cap.read()

        if not ret:
            print("Cannot receive frame")
            break


        # =================================================
        # 鏡像
        # =================================================
        frame = cv2.flip(frame, 1)


        # =================================================
        # 調整畫面
        # =================================================
        frame = cv2.resize(
            frame,
            (FRAME_WIDTH, FRAME_HEIGHT)
        )


        h, w = frame.shape[:2]


        # =================================================
        # 建立第一個目標
        # =================================================
        if not target_initialized:

            target_x, target_y = create_target(w, h)

            target_initialized = True


        # =================================================
        # RGB
        # =================================================
        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        rgb.flags.writeable = False

        results = hands.process(rgb)

        rgb.flags.writeable = True


        # =================================================
        # 遊戲開始
        # =================================================
        if not game_started:

            game_started = True
            game_start_time = time.time()


        # =================================================
        # 計算剩餘時間
        # =================================================
        elapsed = time.time() - game_start_time

        remaining_time = max(
            0,
            GAME_TIME - int(elapsed)
        )


        # =================================================
        # 手部偵測
        # =================================================
        fingertip_found = False

        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:

                # -----------------------------------------
                # MediaPipe Landmark 8 = 食指指尖
                # -----------------------------------------
                fingertip = hand_landmarks.landmark[8]

                raw_x = fingertip.x * w
                raw_y = fingertip.y * h


                # -----------------------------------------
                # EMA 平滑
                # -----------------------------------------
                if smooth_x is None:

                    smooth_x = raw_x
                    smooth_y = raw_y

                else:

                    smooth_x = (
                        SMOOTHING_ALPHA * raw_x
                        +
                        (1 - SMOOTHING_ALPHA) * smooth_x
                    )

                    smooth_y = (
                        SMOOTHING_ALPHA * raw_y
                        +
                        (1 - SMOOTHING_ALPHA) * smooth_y
                    )


                fingertip_found = True


                # -----------------------------------------
                # 繪製手部骨架
                # -----------------------------------------
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )


                # -----------------------------------------
                # 畫出食指指尖
                # -----------------------------------------
                cv2.circle(
                    frame,
                    (
                        int(smooth_x),
                        int(smooth_y)
                    ),
                    10,
                    (255, 255, 0),
                    -1
                )


                # -----------------------------------------
                # 判斷是否碰到目標
                # -----------------------------------------
                if is_inside_target(
                    smooth_x,
                    smooth_y,
                    target_x,
                    target_y
                ):

                    score += 1

                    target_x, target_y = create_target(
                        w,
                        h
                    )


                    # 碰撞後稍微重置平滑
                    smooth_x = None
                    smooth_y = None


                # ------------------------------------------------
                # 目前只使用第一隻手作為游標
                # ------------------------------------------------
                break


        # =====================================================
        # FPS
        # =====================================================
        current_time = time.time()

        delta = current_time - prev_time

        fps = 1 / delta if delta > 0 else 0

        prev_time = current_time


        # =====================================================
        # 目標顏色
        # =====================================================
        target_color = (0, 0, 255)

        # 如果手指碰到目標
        if fingertip_found:

            if is_inside_target(
                smooth_x,
                smooth_y,
                target_x,
                target_y
            ):

                target_color = (0, 255, 0)


        # =====================================================
        # 畫目標
        # =====================================================
        cv2.rectangle(
            frame,
            (target_x, target_y),
            (
                target_x + TARGET_SIZE,
                target_y + TARGET_SIZE
            ),
            target_color,
            4
        )


        # =====================================================
        # UI
        # =====================================================

        # 半透明資訊背景
        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (0, 0),
            (330, 125),
            (0, 0, 0),
            -1
        )

        frame = cv2.addWeighted(
            overlay,
            0.55,
            frame,
            0.45,
            0
        )


        # =====================================================
        # FPS
        # =====================================================
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


        # =====================================================
        # SCORE
        # =====================================================
        cv2.putText(
            frame,
            f"Score: {score}",
            (15, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


        # =====================================================
        # TIME
        # =====================================================
        cv2.putText(
            frame,
            f"Time: {remaining_time}s",
            (15, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )


        # =====================================================
        # 遊戲結束
        # =====================================================
        if remaining_time <= 0:

            cv2.putText(
                frame,
                "GAME OVER",
                (
                    FRAME_WIDTH // 2 - 150,
                    FRAME_HEIGHT // 2
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 0, 255),
                4
            )

            cv2.putText(
                frame,
                f"Final Score: {score}",
                (
                    FRAME_WIDTH // 2 - 150,
                    FRAME_HEIGHT // 2 + 50
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                3
            )


        # =====================================================
        # 顯示
        # =====================================================
        cv2.imshow(
            "Hand Tracking Game",
            frame
        )


        # =====================================================
        # 按鍵
        # =====================================================
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            break


# =========================================================
# 釋放資源
# =========================================================
cap.release()
cv2.destroyAllWindows()