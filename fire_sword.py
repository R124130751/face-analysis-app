import cv2
import numpy as np
import random
import math
import mediapipe as mp
from collections import deque

class MassiveInfernoEngine:
    def __init__(self, history_len=10):
        self.blade_history = deque(maxlen=history_len)
        self.phase = 0.0
        self.particles = []

    def generate_massive_flame_polygons(self, p1, p2, speed_mult=1.0):
        """完全依據手繪輪廓算出的巨型撕裂火焰結構"""
        vec = p2 - p1
        length = np.linalg.norm(vec)
        if length == 0:
            return None, None, None

        u = vec / length              # 刀身方向
        v = np.array([-u[1], u[0]])    # 刀身法向量
        heat_up = np.array([0.0, -1.0])# 向上熱浮力

        # 延伸刀尖火焰 (超越原本的刀尖點 P2)
        tip_extension = u * 70.0 * speed_mult
        p2_ext = p2 + tip_extension

        steps = 180  # 超高採樣點，維持密集鋸齒
        red_top, orange_top, yellow_top = [], [], []
        red_bot, orange_bot, yellow_bot = [], [], []

        for i in range(steps + 1):
            t = i / steps
            curr = p1 + (p2_ext - p1) * t
            
            # 輪廓形狀包絡線：護手處與刀尖收攏，中間與中後段極度膨脹
            envelope = math.sin(t * math.pi)

            # 1. 高頻鋸齒波 + 動態湍流
            saw = (1.0 if i % 2 == 0 else -0.35) * 22.0  # 巨型密集鋸齒刺
            n1 = math.sin(t * 80.0 - self.phase * 6.0) * 15.0
            n2 = math.cos(t * 130.0 + self.phase * 10.0) * 8.0
            turb = (saw + n1 + n2) * envelope

            # 2. 向上飄升的熱浮力 (畫面上方)
            upward_shift = heat_up * (35.0 * envelope * speed_mult + random.uniform(0, 8))

            # 3. 上方火舌高度 (巨型化，最高可達 160+ 像素)
            h_top_red = max(10.0, (110.0 + turb) * speed_mult)
            h_top_ora = max(6.0, h_top_red * 0.6)
            h_top_yel = max(3.0, h_top_red * 0.3)

            # 火舌向前/向上傾斜的張力
            tilt_forward = u * (turb * 0.3)

            red_top.append(curr + v * h_top_red + upward_shift + tilt_forward)
            orange_top.append(curr + v * h_top_ora + upward_shift * 0.7 + tilt_forward * 0.7)
            yellow_top.append(curr + v * h_top_yel + upward_shift * 0.4 + tilt_forward * 0.4)

            # 4. 下方火焰高度 (依圖標註，維持約 40~60 像素的平滑波浪)
            bot_wave = math.sin(t * 15.0 + self.phase * 3.0) * 12.0
            h_bot_red = max(8.0, (45.0 + bot_wave) * envelope * speed_mult)
            h_bot_ora = max(5.0, h_bot_red * 0.6)
            h_bot_yel = max(2.0, h_bot_red * 0.3)

            red_bot.append(curr - v * h_bot_red + upward_shift * 0.2)
            orange_bot.append(curr - v * h_bot_ora + upward_shift * 0.1)
            yellow_bot.append(curr - v * h_bot_yel)

        poly_red = np.array(red_top + red_bot[::-1], dtype=np.int32)
        poly_ora = np.array(orange_top + orange_bot[::-1], dtype=np.int32)
        poly_yel = np.array(yellow_top + yellow_bot[::-1], dtype=np.int32)

        return poly_red, poly_ora, poly_yel

    def emit_high_floating_embers(self, p1, p2, speed_mult=1.0):
        """高空廣域飄散微型火星粒子"""
        count = int(45 * speed_mult)
        vec = p2 - p1
        for _ in range(count):
            alpha = random.uniform(-0.1, 1.1)
            pos = p1 + vec * alpha
            
            # 向高空大幅噴發
            vx = random.uniform(-6, 6)
            vy = random.uniform(-18, -6)  # 強勁向上速度
            
            self.particles.append({
                'x': float(pos[0] + random.uniform(-25, 25)),
                'y': float(pos[1] + random.uniform(-40, 10)), # 起始點偏向上方
                'vx': vx, 'vy': vy,
                'r': random.uniform(0.8, 2.4),
                'life': random.uniform(0.4, 0.9)
            })

    def draw_high_floating_embers(self, canvas):
        alive = []
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= 0.04

            if p['life'] > 0:
                alive.append(p)
                cx, cy = int(p['x']), int(p['y'])
                r = max(1, int(p['r']))

                color = (180, 240, 255) if p['life'] > 0.5 else (0, 90, 220)
                cv2.circle(canvas, (cx, cy), r, color, -1)
        self.particles = alive

def attach_massive_flame_master():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    win_name = 'Massive Raging Inferno Engine'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    engine = MassiveInfernoEngine(history_len=10)
    print("=== 巨型撕裂烈焰引擎 啟動！按下 'q' 退出 ===")

    is_fullscreen = True
    prev_tip = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(rgb_frame)
        
        wrists = []
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            for idx in [15, 16]:
                lm = landmarks[idx]
                if lm.visibility > 0.5:
                    wrists.append(np.array([lm.x * w, lm.y * h]))

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, np.array([0, 130, 60]), np.array([12, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([168, 130, 60]), np.array([180, 255, 255]))
        mask = cv2.bitwise_or(mask1, mask2)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 獨立色彩圖層
        layer_red    = np.zeros_like(frame)
        layer_orange = np.zeros_like(frame)
        layer_yellow = np.zeros_like(frame)
        layer_white  = np.zeros_like(frame)
        layer_embers = np.zeros_like(frame)

        best_cnt = None
        max_area = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 600:
                rect = cv2.minAreaRect(cnt)
                (_, _), (w_rect, h_rect), _ = rect
                length = max(w_rect, h_rect)
                thickness = min(w_rect, h_rect)

                if thickness > 0 and (length / thickness) > 3.0:
                    if area > max_area:
                        max_area = area
                        best_cnt = cnt

        engine.phase += 0.45  # 齒狀快速燃燒

        if best_cnt is not None:
            pts = best_cnt.reshape(-1, 2).astype(np.float32)
            [vx, vy, x0, y0] = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
            u = np.array([vx[0], vy[0]], dtype=np.float32)

            projections = (pts - np.array([x0[0], y0[0]])) @ u
            t_min, t_max = projections.min(), projections.max()

            ep1 = np.array([x0[0], y0[0]]) + t_min * u
            ep2 = np.array([x0[0], y0[0]]) + t_max * u

            P1, P2 = ep1, ep2
            if len(wrists) > 0:
                min_dist_ep1 = min([np.linalg.norm(ep1 - wr) for wr in wrists])
                min_dist_ep2 = min([np.linalg.norm(ep2 - wr) for wr in wrists])

                if min_dist_ep2 < min_dist_ep1:
                    P1, P2 = ep2, ep1

            P1, P2 = P1.astype(int), P2.astype(int)

            speed = 0.0
            if prev_tip is not None:
                speed = np.linalg.norm(P2 - prev_tip)
            prev_tip = P2.copy()

            speed_mult = min(2.5, 1.1 + speed / 20.0)

            # 1. 生成巨型火焰多邊形
            poly_r, poly_o, poly_y = engine.generate_massive_flame_polygons(P1, P2, speed_mult=speed_mult)

            if poly_r is not None:
                # 保留沉穩質感的深赤紅與焦糖橘
                cv2.fillPoly(layer_red, [poly_r], (0, 0, 165))       # 深赤紅
                cv2.fillPoly(layer_orange, [poly_o], (0, 75, 225))   # 飽和琥珀橘
                cv2.fillPoly(layer_yellow, [poly_y], (0, 185, 250))  # 金黃核心

            # 2. 刀身過曝白熱光芯
            cv2.line(layer_white, tuple(P1), tuple(P2), (255, 255, 255), int(7 * speed_mult))
            cv2.circle(layer_white, tuple(P2), int(14 * speed_mult), (255, 255, 255), -1)

            # 3. 高空噴發星火
            engine.emit_high_floating_embers(P1, P2, speed_mult=speed_mult)

        engine.draw_high_floating_embers(layer_embers)

        # --- 巨型化專用高斯模糊核（完全無硬邊，質感過渡） ---
        blur_red    = cv2.GaussianBlur(layer_red, (141, 141), 0)    # 超廣角深紅擴散
        blur_orange = cv2.GaussianBlur(layer_orange, (65, 65), 0)   # 橘色氣體羽化
        blur_yellow = cv2.GaussianBlur(layer_yellow, (29, 29), 0)   # 核心金黃
        blur_white  = cv2.GaussianBlur(layer_white, (11, 11), 0)
        blur_embers = cv2.GaussianBlur(layer_embers, (3, 3), 0)

        # 熱殘影稍微向上（-y）微移
        M_up = np.float32([[1, 0, 0], [0, 1, -12]])
        blur_red = cv2.warpAffine(blur_red, M_up, (w, h))

        # 4. 融合火焰主體
        flame_body = cv2.add(blur_red, blur_orange)
        flame_body = cv2.add(flame_body, blur_yellow)
        flame_body = cv2.add(flame_body, blur_white)
        flame_body = cv2.add(flame_body, blur_embers)

        # 5. 超廣域強烈環境光暈
        glow_layer = cv2.GaussianBlur(flame_body, (135, 135), 0)
        glow_layer = cv2.convertScaleAbs(glow_layer, alpha=2.2, beta=0)

        # 6. 畫面疊加
        output = cv2.add(frame, glow_layer)
        output = cv2.add(output, flame_body)

        cv2.imshow(win_name, output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('f'):
            is_fullscreen = not is_fullscreen
            flag = cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL
            cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, flag)

    pose.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    attach_massive_flame_master()