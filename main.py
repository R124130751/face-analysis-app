import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
import math
import re
from datetime import datetime
from PIL import Image, ImageTk, ImageDraw, ImageFont
import matplotlib.pyplot as plt

import illustrations as ill  # 同資料夾的 illustrations.py

# 設定 Matplotlib 中文字型
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# 1. 生物力學與關節中立位分析引擎
# ==========================================
class PostureAnalyzer:
    @staticmethod
    def calculate_angle_2d(a, b, c):
        ba = np.array([a[0] - b[0], a[1] - b[1]])
        bc = np.array([c[0] - b[0], c[1] - b[1]])
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)

    @staticmethod
    def calculate_vertical_angle(p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return math.degrees(math.atan2(abs(dx), abs(dy)))

    @staticmethod
    def calculate_horizontal_angle(p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return math.degrees(math.atan2(abs(dy), abs(dx)))

    def analyze(self, landmarks, image_w, image_h):
        pts = {idx: (lm.x * image_w, lm.y * image_h) for idx, lm in enumerate(landmarks)}

        ear_avg = ((pts[7][0] + pts[8][0]) / 2, (pts[7][1] + pts[8][1]) / 2)
        shoulder_avg = ((pts[11][0] + pts[12][0]) / 2, (pts[11][1] + pts[12][1]) / 2)
        hip_avg = ((pts[23][0] + pts[24][0]) / 2, (pts[23][1] + pts[24][1]) / 2)
        knee_avg = ((pts[25][0] + pts[26][0]) / 2, (pts[25][1] + pts[26][1]) / 2)

        fha_diff = abs(self.calculate_vertical_angle(ear_avg, shoulder_avg) - 0.0)
        shoulder_diff = abs(self.calculate_horizontal_angle(pts[11], pts[12]) - 0.0)
        spine_diff = abs(self.calculate_vertical_angle(shoulder_avg, hip_avg) - 0.0)
        pelvis_diff = abs(self.calculate_horizontal_angle(pts[23], pts[24]) - 0.0)
        knee_diff = abs(self.calculate_vertical_angle(hip_avg, knee_avg) - 0.0)

        def get_score(diff, max_penalty=15.0):
            return max(30, min(100, round(100 - (diff / max_penalty) * 40)))

        scores = {
            "頭頸中立": get_score(fha_diff, 12.0),
            "肩胛盂肱關節": get_score(shoulder_diff, 8.0),
            "脊柱胸廓排列": get_score(spine_diff, 10.0),
            "骨盆體態": get_score(pelvis_diff, 8.0),
            "下肢關節力線": get_score(knee_diff, 10.0)
        }

        overall_score = round(sum(scores.values()) / len(scores))
        risk_index = round(100 - overall_score)

        details = [
            {"view": "側面", "name": "頭部姿勢", "val": round(fha_diff, 1), "unit": "°",
             "status": "警惕" if fha_diff > 5.0 else "維持", "range": (11, 30),
             "factors": ["1. 長時間使用手機或平板", "2. 長時間使用電腦", "3. 使用過高的枕頭"]},
            {"view": "正面", "name": "肩膀高低差", "val": round(shoulder_diff, 1), "unit": "°",
             "status": "警惕" if shoulder_diff > 2.5 else "維持", "range": (0, 4),
             "factors": ["1. 習慣單肩背包", "2. 習慣單手抱重物", "3. 坐姿歪斜、慣用側傾斜"]},
            {"view": "側面", "name": "圓肩 (前拉)", "val": 14.0, "unit": "°", "status": "警惕", "range": (5, 26),
             "factors": ["1. 胸肌過度緊繃", "2. 背肌無力拉引", "3. 打字手肘無支撐"]},
            {"view": "側面", "name": "膝關節屈曲", "val": round(knee_diff, 1), "unit": "°",
             "status": "警惕" if knee_diff > 5.0 else "維持", "range": (0, 11),
             "factors": ["1. 大腿後側肌肉緊繃", "2. 股四頭肌力量失衡"]},
            {"view": "正面", "name": "HKA-角度", "val": 0.0, "unit": "°", "status": "維持", "range": (0, 7),
             "factors": ["1. 經常盤腿坐習慣", "2. 脊柱內/外八步態習慣"]},
            {"view": "背面", "name": "脊椎側彎程度", "val": round(spine_diff, 1), "unit": "°",
             "status": "警惕" if spine_diff > 3.0 else "維持", "range": (0, 25),
             "factors": ["1. 翹腳坐姿習慣", "2. 姿勢性單側發力"]},
            {"view": "側面", "name": "胸椎後凸", "val": 46.0, "unit": "°", "status": "警惕", "range": (36, 52),
             "factors": ["1. 長時間低頭彎腰", "2. 核心肌群缺乏支撐"]},
            {"view": "側面", "name": "腰椎前凸", "val": 38.0, "unit": "°", "status": "維持", "range": (35, 49),
             "factors": ["1. 挺肚子站立習慣", "2. 腹肌無力、髂腰肌緊繃"]},
            {"view": "背面", "name": "骨盆傾斜", "val": round(pelvis_diff, 1), "unit": "°",
             "status": "警惕" if pelvis_diff > 2.5 else "維持", "range": (0, 3),
             "factors": ["1. 站立時習慣重心在單側", "2. 生產後骨盆未復位"]},
            {"view": "側面", "name": "骨盆後傾", "val": 5.0, "unit": "°", "status": "警惕", "range": (0, 8),
             "factors": ["1. 坐姿時習慣臀部前移(癱坐)", "2. 站立時膝蓋過度挺直"]}
        ]

        metrics = {
            "shoulder_diff": shoulder_diff, "pelvis_diff": pelvis_diff,
            "hka_l": 0.0, "hka_r": 0.0,
            "head_forward": fha_diff, "kyphosis": 46.0, "lordosis": 38.0,
            "pelvic_tilt": 5.0, "knee_flex": knee_diff,
            "round_shoulder": 14.0, "scoliosis": spine_diff,
        }

        return {"overall_score": overall_score, "risk_index": risk_index, "sub_scores": scores,
                "details": details, "raw_pts": pts, "metrics": metrics}


# ==========================================
# 2. 報告插圖引擎（總覽骨架圖 / 雷達圖）
# ==========================================
class ReportGenerator:
    @staticmethod
    def draw_skeleton_overview(metrics):
        return ill.draw_skeleton_overview(metrics)

    @staticmethod
    def create_radar_chart(scores):
        import io
        categories = list(scores.keys())
        values = list(scores.values())
        N = len(categories)
        angles = [n / float(N) * 2 * math.pi for n in range(N)]
        values += values[:1]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(3.6, 3.6), subplot_kw=dict(polar=True))
        plt.subplots_adjust(top=0.85, bottom=0.15)
        ax.plot(angles, values, color='#2563EB', linewidth=2)
        ax.fill(angles, values, color='#3B82F6', alpha=0.3)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=9)
        plt.yticks([20, 40, 60, 80, 100], ["20", "40", "60", "80", "100"], color="grey", size=7)
        plt.ylim(0, 100)
        ax.set_title("體態中立位綜合雷達圖", fontsize=10, pad=12, weight='bold')

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return Image.open(buf)


# ==========================================
# 2b. 自繪滑桿元件（顯示角度落在正常/警惕區間的位置）
# ==========================================
class RangeSlider(tk.Canvas):
    def __init__(self, parent, vmin, vmax, value, warn, width=250, height=40, **kw):
        super().__init__(parent, width=width, height=height, bg="#F8FAFC", highlightthickness=0, **kw)
        self._draw(vmin, vmax, value, warn, width, height)

    def _draw(self, vmin, vmax, value, warn, width, height):
        margin = 18
        x0, x1 = margin, width - margin
        track_y = height * 0.42
        self.create_line(x0, track_y, x1, track_y, fill="#E2E8F0", width=5, capstyle=tk.ROUND)

        rng = (vmax - vmin) or 1
        ratio = max(0.0, min(1.0, (value - vmin) / rng))
        px = x0 + (x1 - x0) * ratio
        color = "#E11D48" if warn else "#10B981"
        self.create_line(x0, track_y, px, track_y, fill=color, width=5, capstyle=tk.ROUND)
        self.create_oval(px - 6.5, track_y - 6.5, px + 6.5, track_y + 6.5,
                          fill="white", outline=color, width=2.4)

        for i in range(4):
            tx = x0 + (x1 - x0) * i / 3
            tv = vmin + rng * i / 3
            self.create_line(tx, track_y + 7, tx, track_y + 11, fill="#CBD5E1")
            self.create_text(tx, track_y + 20, text=f"{tv:.0f}°", font=("Microsoft JhengHei", 7), fill="#94A3B8")


# ==========================================
# 3. 診斷報告視窗
# ==========================================
class PostureReportWindow(tk.Toplevel):
    def __init__(self, parent, user_info, analysis_result, captured_photos):
        super().__init__(parent)
        self.title("MotiPhysio 體態骨架分析診斷報告")
        self.geometry("1180x860")
        self.configure(bg="#F8FAFC")

        self.user_info = user_info
        self.analysis = analysis_result
        self.captured_photos = captured_photos
        self._img_refs = []  # 避免 PhotoImage 被回收

        self.build_ui()

    # ---------- 小工具 ----------
    def _photo(self, pil_img, size=None):
        if size:
            pil_img = pil_img.resize(size, Image.LANCZOS)
        photo = ImageTk.PhotoImage(pil_img)
        self._img_refs.append(photo)
        return photo

    def build_ui(self):
        top_bar = tk.Frame(self, bg="#0284C7", height=42)
        top_bar.pack(fill="x")
        tk.Label(top_bar, text="虎鐵器械皮拉提斯 | 體態評估報告", font=("Microsoft JhengHei", 12, "bold"),
                 fg="#FFFFFF", bg="#0284C7").pack(side="left", padx=15, pady=8)
        tk.Label(top_bar, text="Moti Physio", font=("Microsoft JhengHei", 12, "italic", "bold"),
                 fg="#FFFFFF", bg="#0284C7").pack(side="right", padx=15, pady=8)

        main_nb = ttk.Notebook(self)
        main_nb.pack(fill="both", expand=True, padx=12, pady=10)

        self._build_page1(main_nb)
        self._build_page2(main_nb)
        self._build_page3(main_nb)

    # ---------------- Page 1: 骨架總覽報告表 ----------------
    def _build_page1(self, main_nb):
        p1 = ttk.Frame(main_nb)
        main_nb.add(p1, text="📋 1/2 頁：骨架總覽報告表")

        p1_left = tk.Frame(p1, bg="#FFFFFF", relief="solid", bd=1)
        p1_left.pack(side="left", fill="both", expand=True, padx=(10, 5), pady=10)

        skel_img = ReportGenerator.draw_skeleton_overview(self.analysis["metrics"])
        lbl_skel = tk.Label(p1_left, image=self._photo(skel_img), bg="#FFFFFF")
        lbl_skel.pack(fill="both", expand=True, padx=10, pady=10)

        p1_right = tk.Frame(p1, bg="#F8FAFC", width=380)
        p1_right.pack(side="right", fill="y", padx=(5, 10), pady=10)
        p1_right.pack_propagate(False)

        # 個人資訊
        info_box = tk.LabelFrame(p1_right, text=" 骨架報告表 ", font=("Microsoft JhengHei", 11, "bold"),
                                  bg="#FFFFFF", fg="#0284C7", padx=12, pady=10)
        info_box.pack(fill="x", pady=(0, 10))

        curr_time = datetime.now().strftime("%Y/%m/%d (PM %I:%M)")
        tk.Label(info_box, text=f"檢測日期：{curr_time}", font=("Microsoft JhengHei", 9),
                 bg="#FFFFFF", fg="#64748B").pack(anchor="w")
        tk.Label(info_box, text=f"姓名/暱稱：{self.user_info.get('name', '林世瑋')}",
                 font=("Microsoft JhengHei", 11, "bold"), bg="#FFFFFF", fg="#1E293B").pack(anchor="w", pady=3)
        tk.Label(info_box, text=f"性別：{self.user_info.get('gender', '男性')}   |   生日：{self.user_info.get('birthday', '1991.03.25')}",
                 font=("Microsoft JhengHei", 9.5), bg="#FFFFFF", fg="#475569").pack(anchor="w")

        # 風險指數（含刻度與指針）
        risk_box = tk.LabelFrame(p1_right, text=" 體態失衡風險評估 ", font=("Microsoft JhengHei", 10, "bold"),
                                  bg="#FFFFFF", fg="#334155", padx=12, pady=10)
        risk_box.pack(fill="x", pady=(0, 10))

        r_idx = self.analysis["risk_index"]
        r_str = "危險" if r_idx > 66 else ("警惕" if r_idx > 33 else "維持")
        r_color = "#E11D48" if r_idx > 66 else ("#F59E0B" if r_idx > 33 else "#10B981")

        f_risk = tk.Frame(risk_box, bg="#FFFFFF")
        f_risk.pack(fill="x")
        tk.Label(f_risk, text="風險指數", font=("Microsoft JhengHei", 10), bg="#FFFFFF", fg="#475569").pack(side="left")
        tk.Label(f_risk, text=f"{r_idx}", font=("Microsoft JhengHei", 26, "bold"), fg=r_color, bg="#FFFFFF").pack(side="left", padx=12)
        tk.Label(f_risk, text=r_str, font=("Microsoft JhengHei", 10, "bold"), fg="#FFFFFF", bg=r_color, padx=8, pady=2).pack(side="left")

        gauge = tk.Canvas(risk_box, width=300, height=42, bg="#FFFFFF", highlightthickness=0)
        gauge.pack(fill="x", pady=6)
        seg_colors = ["#10B981", "#F59E0B", "#E11D48"]
        seg_labels = ["維持", "警惕", "危險"]
        for i, c in enumerate(seg_colors):
            gauge.create_rectangle(i * 100, 6, i * 100 + 98, 16, fill=c, width=0)
            gauge.create_text(i * 100 + 49, 30, text=seg_labels[i], font=("Microsoft JhengHei", 8), fill="#94A3B8")
        px = max(4, min(296, r_idx / 100 * 300))
        gauge.create_polygon(px - 6, 0, px + 6, 0, px, 8, fill="#1E293B")

        # 三角度快照
        photo_box = tk.LabelFrame(p1_right, text=" 檢測角度影像快照 ", font=("Microsoft JhengHei", 10, "bold"),
                                   bg="#FFFFFF", fg="#334155", padx=8, pady=8)
        photo_box.pack(fill="both", expand=True)

        p_frame = tk.Frame(photo_box, bg="#FFFFFF")
        p_frame.pack(fill="both", expand=True)

        for view_k in ["正面", "側面", "反面"]:
            sub_f = tk.Frame(p_frame, bg="#FFFFFF")
            sub_f.pack(side="left", fill="both", expand=True, padx=2)

            if self.captured_photos.get(view_k) is not None:
                img_cv = cv2.resize(self.captured_photos[view_k], (105, 145))
                img_p = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
                lbl_p = tk.Label(sub_f, image=self._photo(img_p), bg="#FFFFFF")
                lbl_p.pack()
            else:
                tk.Label(sub_f, text=f"{view_k}\n未拍攝", font=("Microsoft JhengHei", 9),
                          bg="#F1F5F9", width=11, height=8).pack()

            tk.Label(sub_f, text=view_k, font=("Microsoft JhengHei", 9), bg="#FFFFFF", fg="#64748B").pack(pady=2)

    # ---------------- Page 2: 10 大關節卡片 ----------------
    def _build_page2(self, main_nb):
        p2 = ttk.Frame(main_nb)
        main_nb.add(p2, text="🔍 2/2 頁：關節肌肉影響詳細分析")

        canvas2 = tk.Canvas(p2, bg="#F1F5F9")
        scrollbar2 = ttk.Scrollbar(p2, orient="vertical", command=canvas2.yview)
        scroll_frame2 = tk.Frame(canvas2, bg="#F1F5F9")

        scroll_frame2.bind("<Configure>", lambda e: canvas2.configure(scrollregion=canvas2.bbox("all")))
        canvas2.create_window((0, 0), window=scroll_frame2, anchor="nw")
        canvas2.configure(yscrollcommand=scrollbar2.set)

        canvas2.pack(side="left", fill="both", expand=True)
        scrollbar2.pack(side="right", fill="y")

        for idx, item in enumerate(self.analysis["details"]):
            r_idx, c_idx = idx // 2, idx % 2
            card = tk.Frame(scroll_frame2, bg="#FFFFFF", relief="solid", bd=1, width=560, height=232)
            card.grid(row=r_idx, column=c_idx, padx=10, pady=8, sticky="nsew")
            card.grid_propagate(False)
            self._build_joint_card(card, item)

    def _build_joint_card(self, card, item):
        warn = item['status'] == "警惕"

        # 標題列
        h_frame = tk.Frame(card, bg="#FFFFFF")
        h_frame.pack(fill="x", padx=12, pady=(10, 2))
        tk.Label(h_frame, text=f"[{item['view']}] {item['name']}", font=("Microsoft JhengHei", 11, "bold"),
                 fg="#0F172A", bg="#FFFFFF").pack(side="left")
        st_color = "#E11D48" if warn else "#10B981"
        tk.Label(h_frame, text=item['status'], font=("Microsoft JhengHei", 8.5, "bold"), fg="#FFFFFF",
                 bg=st_color, padx=7, pady=1).pack(side="right")

        # 滑桿（失衡程度區間）
        s_frame = tk.Frame(card, bg="#FFFFFF")
        s_frame.pack(fill="x", padx=12)
        tk.Label(s_frame, text="失衡程度", font=("Microsoft JhengHei", 8.5), fg="#64748B", bg="#FFFFFF").pack(anchor="w")
        vmin, vmax = item.get("range", (0, max(item['val'] * 2, 10)))
        slider = RangeSlider(s_frame, vmin, vmax, item['val'], warn, width=300, height=40)
        slider.pack(anchor="w")
        tk.Label(s_frame, text=f"{item['val']}{item['unit']}", font=("Microsoft JhengHei", 12, "bold"),
                 fg=st_color, bg="#FFFFFF").place(in_=slider, relx=1.0, x=-4, y=-4, anchor="ne")

        # 主體：小示意圖 + 肌肉分佈圖 + 影響因素
        body = tk.Frame(card, bg="#FFFFFF")
        body.pack(fill="both", expand=True, padx=10, pady=(4, 8))

        icon_img = ill.draw_joint_icon(item['name'])
        tk.Label(body, image=self._photo(icon_img, size=(58, 78)), bg="#FFFFFF").pack(side="left", padx=(2, 4))

        muscle_img = ill.draw_muscle_pair(item['name'])
        tk.Label(body, image=self._photo(muscle_img, size=(140, 118)), bg="#FFFFFF").pack(side="left", padx=4)

        fact_frame = tk.Frame(body, bg="#F8FAFC")
        fact_frame.pack(side="left", fill="both", expand=True, padx=(6, 2))
        tk.Label(fact_frame, text="影響因素：", font=("Microsoft JhengHei", 8.5, "bold"), fg="#0284C7",
                 bg="#F8FAFC").pack(anchor="w", padx=4, pady=(4, 0))
        for fc in item["factors"]:
            tk.Label(fact_frame, text=fc, font=("Microsoft JhengHei", 8.3), fg="#334155", bg="#F8FAFC",
                     wraplength=170, justify="left").pack(anchor="w", padx=10)

    # ---------------- Page 3: 雷達圖 ----------------
    def _build_page3(self, main_nb):
        p3 = ttk.Frame(main_nb)
        main_nb.add(p3, text="📊 雷達圖分析")
        radar_img = ReportGenerator.create_radar_chart(self.analysis["sub_scores"])
        lbl_r = tk.Label(p3, image=self._photo(radar_img), bg="#FFFFFF")
        lbl_r.pack(expand=True)


# ==========================================
# 4. 主應用程式
# ==========================================
class MotiPostureApp:
    def __init__(self, window):
        self.window = window
        self.window.title("MotiPosture 體態分析系統")
        self.window.geometry("1150x720")
        self.window.configure(bg="#E2E8F0")

        self.cap = cv2.VideoCapture(0)
        self.analyzer = PostureAnalyzer()

        self.latest_frame_bgr = None
        self.latest_landmarks = None
        self.selected_view = "正面"

        self.captured_photos = {"正面": None, "側面": None, "反面": None}
        self.countdown_seconds = 0
        self.is_counting = False

        self.build_minimalist_ui()
        self.update_frame()

    def build_minimalist_ui(self):
        header = tk.Frame(self.window, bg="#1E293B", height=40)
        header.pack(fill="x", side="top")

        title_lbl = tk.Label(header, text="MotiPosture 體態分析系統", font=("Microsoft JhengHei", 11, "bold"),
                             fg="#F8FAFC", bg="#1E293B")
        title_lbl.pack(side="left", padx=15, pady=8)

        main_layout = tk.Frame(self.window, bg="#E2E8F0")
        main_layout.pack(fill="both", expand=True, padx=12, pady=12)

        left_column = tk.Frame(main_layout, bg="#E2E8F0", width=310)
        left_column.pack(side="left", fill="y", padx=(0, 10))
        left_column.pack_propagate(False)

        card_info = tk.LabelFrame(left_column, text=" 個人基本資料 ", font=("Microsoft JhengHei", 10, "bold"),
                                  bg="#FFFFFF", fg="#334155", relief="solid", bd=1, padx=12, pady=10)
        card_info.pack(fill="x", pady=(0, 10))

        fields = [("姓名", "name"), ("生日", "birthday"), ("性別", "gender"), ("郵件信箱", "email")]
        self.entries = {}

        for idx, (label_text, field_key) in enumerate(fields):
            lbl = tk.Label(card_info, text=f"{label_text}：", font=("Microsoft JhengHei", 10), bg="#FFFFFF", fg="#475569")
            lbl.grid(row=idx, column=0, sticky="w", pady=6)

            entry = tk.Entry(card_info, font=("Microsoft JhengHei", 10), bg="#F8FAFC", relief="solid", bd=1)
            entry.grid(row=idx, column=1, sticky="ew", pady=6, padx=(5, 0))
            self.entries[field_key] = entry

        card_info.columnconfigure(1, weight=1)

        card_control = tk.LabelFrame(left_column, text=" 檢測角度與控制 ", font=("Microsoft JhengHei", 10, "bold"),
                                     bg="#FFFFFF", fg="#334155", relief="solid", bd=1, padx=12, pady=12)
        card_control.pack(fill="both", expand=True)

        tk.Label(card_control, text="選擇檢測角度：", font=("Microsoft JhengHei", 9), bg="#FFFFFF", fg="#64748B").pack(anchor="w", pady=(0, 5))

        self.view_btns = {}
        for view_name in ["正面", "側面", "反面"]:
            btn = tk.Button(card_control, text=f"{view_name}(點擊 5 秒倒數攝影)", font=("Microsoft JhengHei", 10),
                            bg="#F1F5F9", fg="#334155", activebackground="#CBD5E1", relief="flat", bd=1,
                            command=lambda v=view_name: self.start_angle_countdown(v))
            btn.pack(fill="x", pady=4, ipady=4)
            self.view_btns[view_name] = btn

        self.set_view_mode("正面")

        btn_box = tk.Frame(card_control, bg="#FFFFFF")
        btn_box.pack(fill="x", side="bottom", pady=(10, 0))

        clear_btn = tk.Button(btn_box, text="清除鈕", font=("Microsoft JhengHei", 10), bg="#E2E8F0", fg="#475569",
                              relief="flat", command=self.clear_fields)
        clear_btn.pack(fill="x", pady=(0, 6), ipady=3)

        finish_btn = tk.Button(btn_box, text="完成鈕\n(生成完整診斷報告)", font=("Microsoft JhengHei", 10, "bold"),
                               bg="#2563EB", fg="#FFFFFF", activebackground="#1D4ED8", relief="flat",
                               command=self.finish_and_send)
        finish_btn.pack(fill="x", ipady=4)

        right_column = tk.Frame(main_layout, bg="#FFFFFF", relief="solid", bd=1)
        right_column.pack(side="right", fill="both", expand=True)

        self.video_label = tk.Label(right_column, bg="#0F172A")
        self.video_label.pack(fill="both", expand=True, padx=2, pady=2)

    def set_view_mode(self, mode_name):
        self.selected_view = mode_name
        for v_name, btn in self.view_btns.items():
            if v_name == mode_name:
                btn.config(bg="#3B82F6", fg="#FFFFFF")
            else:
                btn.config(bg="#F1F5F9", fg="#334155")

    def start_angle_countdown(self, view_name):
        if self.is_counting:
            return
        self.set_view_mode(view_name)
        self.countdown_seconds = 5
        self.is_counting = True
        self.run_countdown()

    def run_countdown(self):
        if self.countdown_seconds > 0:
            self.window.after(1000, self.decrement_countdown)
        else:
            self.is_counting = False
            if self.latest_frame_bgr is not None:
                self.captured_photos[self.selected_view] = self.latest_frame_bgr.copy()
            messagebox.showinfo("拍攝成功", f"【{self.selected_view}】角度畫面已快照擷取！")

    def decrement_countdown(self):
        self.countdown_seconds -= 1
        self.run_countdown()

    def validate_inputs(self):
        name = self.entries["name"].get().strip()
        birthday = self.entries["birthday"].get().strip()
        gender = self.entries["gender"].get().strip()
        email = self.entries["email"].get().strip()

        errors = []
        if not name:
            errors.append("• 姓名不可為空白")

        if not birthday:
            errors.append("• 生日不可為空白")
        else:
            valid_d = False
            for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
                try:
                    datetime.strptime(birthday, fmt)
                    valid_d = True
                    break
                except ValueError:
                    pass
            if not valid_d:
                errors.append("• 生日格式錯誤 (例: 1991-03-25)")

        if not gender or gender.upper() not in ["男", "女", "M", "F", "MALE", "FEMALE"]:
            errors.append("• 性別格式錯誤 (請填寫 '男' 或 '女')")

        email_pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
        if not email or not re.match(email_pattern, email):
            errors.append("• 電子信箱格式錯誤 (例: name@example.com)")

        if errors:
            messagebox.showerror("格式驗證提醒", "\n".join(errors))
            return False
        return True

    def clear_fields(self):
        for entry in self.entries.values():
            entry.delete(0, tk.END)
        self.set_view_mode("正面")
        self.captured_photos = {"正面": None, "側面": None, "反面": None}

    def finish_and_send(self):
        if not self.validate_inputs():
            return

        if self.latest_frame_bgr is None or self.latest_landmarks is None:
            messagebox.showwarning("提示", "未擷取到有效鏡頭畫面！")
            return

        h, w, _ = self.latest_frame_bgr.shape
        analysis_result = self.analyzer.analyze(self.latest_landmarks, w, h)

        user_info = {
            "name": self.entries["name"].get().strip(),
            "birthday": self.entries["birthday"].get().strip(),
            "gender": self.entries["gender"].get().strip(),
            "email": self.entries["email"].get().strip()
        }

        PostureReportWindow(self.window, user_info, analysis_result, self.captured_photos)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            self.latest_frame_bgr = frame.copy()
            rendered_frame = frame.copy()

            class FakePoint:
                def __init__(self, x, y):
                    self.x = x
                    self.y = y

            self.latest_landmarks = [
                FakePoint(0.5, 0.2), FakePoint(0.0, 0.0), FakePoint(0.0, 0.0), FakePoint(0.0, 0.0),
                FakePoint(0.0, 0.0), FakePoint(0.0, 0.0), FakePoint(0.0, 0.0), FakePoint(0.48, 0.22),
                FakePoint(0.52, 0.22), FakePoint(0.0, 0.0), FakePoint(0.0, 0.0), FakePoint(0.44, 0.40),
                FakePoint(0.56, 0.42), FakePoint(0.0, 0.0), FakePoint(0.0, 0.0), FakePoint(0.0, 0.0),
                FakePoint(0.0, 0.0), FakePoint(0.0, 0.0), FakePoint(0.0, 0.0), FakePoint(0.0, 0.0),
                FakePoint(0.0, 0.0), FakePoint(0.0, 0.0), FakePoint(0.0, 0.0), FakePoint(0.45, 0.65),
                FakePoint(0.55, 0.66), FakePoint(0.45, 0.80), FakePoint(0.55, 0.80), FakePoint(0.45, 0.95), FakePoint(0.55, 0.95)
            ]

            for lm in [self.latest_landmarks[i] for i in [0, 7, 8, 11, 12, 23, 24]]:
                cv2.circle(rendered_frame, (int(lm.x * w), int(lm.y * h)), 5, (0, 255, 0), -1)

            cv2.putText(rendered_frame, f"Mode: {self.selected_view}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            img = Image.fromarray(cv2.cvtColor(rendered_frame, cv2.COLOR_BGR2RGB))

            if self.is_counting and self.countdown_seconds > 0:
                draw = ImageDraw.Draw(img)
                try:
                    font_large = ImageFont.truetype("msjh.ttc", 90)
                    font_sub = ImageFont.truetype("msjh.ttc", 22)
                except Exception:
                    font_large = ImageFont.load_default()
                    font_sub = ImageFont.load_default()

                draw.rectangle([(w // 2 - 160, h // 2 - 100), (w // 2 + 160, h // 2 + 80)], fill=(0, 0, 0, 180))
                draw.text((w // 2 - 25, h // 2 - 70), str(self.countdown_seconds), fill=(255, 215, 0), font=font_large)
                draw.text((w // 2 - 120, h // 2 + 30), f"【{self.selected_view}】倒數拍攝中...", fill=(255, 255, 255), font=font_sub)

            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.window.after(30, self.update_frame)


if __name__ == "__main__":
    root = tk.Tk()
    app = MotiPostureApp(root)
    root.mainloop()