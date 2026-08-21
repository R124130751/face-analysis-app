import cv2
import mediapipe as mp
import time


# =========================================================
# MediaPipe Objectron
# =========================================================
mp_drawing = mp.solutions.drawing_utils
mp_objectron = mp.solutions.objectron


# =========================================================
# Camera Settings
# =========================================================
CAMERA_ID = 0

FRAME_WIDTH = 960
FRAME_HEIGHT = 540

WINDOW_NAME = "MediaPipe 3D Shoe Detection"


# =========================================================
# FPS Smoothing
# =========================================================
fps = 0.0
prev_time = time.time()

FPS_ALPHA = 0.1


# =========================================================
# Camera
# =========================================================
cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():
    print("[ERROR] Cannot open camera")
    raise SystemExit


cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# 降低 Buffer，減少延遲
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


# =========================================================
# MediaPipe Objectron
# =========================================================
with mp_objectron.Objectron(

    static_image_mode=False,

    max_num_objects=5,

    min_detection_confidence=0.5,

    min_tracking_confidence=0.7,

    model_name="Shoe"

) as objectron:

    print("===================================")
    print(" MediaPipe 3D Shoe Detection")
    print("===================================")
    print("Press Q or ESC to exit")


    while True:

        # =================================================
        # Capture
        # =================================================
        ret, frame = cap.read()

        if not ret:
            print("[ERROR] Cannot receive frame")
            break


        # =================================================
        # Mirror
        # =================================================
        frame = cv2.flip(frame, 1)


        # =================================================
        # Resize
        # =================================================
        frame = cv2.resize(
            frame,
            (FRAME_WIDTH, FRAME_HEIGHT)
        )


        # =================================================
        # BGR → RGB
        # =================================================
        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        rgb.flags.writeable = False


        # =================================================
        # Object Detection
        # =================================================
        results = objectron.process(rgb)


        rgb.flags.writeable = True


        # =================================================
        # Object Count
        # =================================================
        object_count = 0


        if results.detected_objects:

            object_count = len(
                results.detected_objects
            )


            # =============================================
            # Process each detected shoe
            # =============================================
            for index, detected_object in enumerate(
                results.detected_objects
            ):

                # -----------------------------------------
                # 2D Bounding Box / Landmark
                # -----------------------------------------
                mp_drawing.draw_landmarks(

                    frame,

                    detected_object.landmarks_2d,

                    mp_objectron.BOX_CONNECTIONS
                )


                # -----------------------------------------
                # 3D Axis
                # -----------------------------------------
                mp_drawing.draw_axis(

                    frame,

                    detected_object.rotation,

                    detected_object.translation
                )


                # -----------------------------------------
                # Object Label
                # -----------------------------------------
                # 取得 2D landmarks 中心位置
                landmarks = (
                    detected_object.landmarks_2d.landmark
                )

                if landmarks:

                    xs = [
                        int(lm.x * FRAME_WIDTH)
                        for lm in landmarks
                    ]

                    ys = [
                        int(lm.y * FRAME_HEIGHT)
                        for lm in landmarks
                    ]

                    center_x = sum(xs) // len(xs)
                    center_y = sum(ys) // len(ys)


                    cv2.putText(

                        frame,

                        f"Shoe #{index + 1}",

                        (
                            center_x,
                            center_y
                        ),

                        cv2.FONT_HERSHEY_SIMPLEX,

                        0.65,

                        (0, 255, 0),

                        2
                    )


        # =================================================
        # FPS Calculation
        # =================================================
        current_time = time.time()

        delta_time = current_time - prev_time

        prev_time = current_time


        if delta_time > 0:

            current_fps = 1.0 / delta_time

            fps = (
                FPS_ALPHA * current_fps
                +
                (1 - FPS_ALPHA) * fps
            )


        # =================================================
        # UI Background
        # =================================================
        overlay = frame.copy()

        cv2.rectangle(

            overlay,

            (0, 0),

            (350, 120),

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


        # =================================================
        # FPS
        # =================================================
        cv2.putText(

            frame,

            f"FPS: {fps:.1f}",

            (15, 30),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.75,

            (0, 255, 0),

            2
        )


        # =================================================
        # Object Count
        # =================================================
        cv2.putText(

            frame,

            f"Shoes: {object_count}",

            (15, 65),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.75,

            (0, 255, 255),

            2
        )


        # =================================================
        # Model
        # =================================================
        cv2.putText(

            frame,

            "Model: MediaPipe Objectron",

            (15, 100),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            (255, 255, 255),

            1
        )


        # =================================================
        # Display
        # =================================================
        cv2.imshow(

            WINDOW_NAME,

            frame
        )


        # =================================================
        # Keyboard
        # =================================================
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:

            break


# =========================================================
# Cleanup
# =========================================================
cap.release()

cv2.destroyAllWindows()

print("Camera released.")
print("Program terminated.")