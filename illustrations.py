import math
import os
import platform
from PIL import Image, ImageDraw, ImageFont

def get_font(size=14, bold=False):
    """跨平台中文字型載入器"""
    system = platform.system()
    fonts = []
    
    if system == "Windows":
        fonts = ["msjhbd.ttc" if bold else "msjh.ttc", "simhei.ttf", "kaiu.ttf"]
    elif system == "Darwin": # macOS
        fonts = ["/System/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/STHeiti Light.ttc"]
    else: # Linux / Docker
        fonts = ["/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", "DejaVuSans.ttf"]

    for font_path in fonts:
        try:
            return ImageFont.truetype(font_path, size)
        except OSError:
            continue
            
    # 若系統無指定字型則回傳預設字型
    return ImageFont.load_default()

def draw_skeleton_overview(metrics):
    """繪製第一頁左側骨架圖（精準對齊、字型強化、適應解析度）"""
    width, height = 550, 680
    img = Image.new("RGB", (width, height), "#0F172A")
    draw = ImageDraw.Draw(img)

    font_title = get_font(16, bold=True)
    font_label = get_font(12, bold=True)
    font_sub = get_font(11)
    font_val = get_font(13, bold=True)

    # 標題
    draw.text((20, 15), "骨架體態排列與核心力線分析", fill="#38BDF8", font=font_title)
    draw.line([(20, 42), (width - 20, 42)], fill="#334155", width=1)

    views = [
        ("正面 (Anterior)", 110, metrics.get("shoulder_diff", 0.0), "肩高低差"),
        ("側面 (Sagittal)", 275, metrics.get("head_forward", 0.0), "頭前傾"),
        ("背面 (Posterior)", 440, metrics.get("scoliosis", 0.0), "脊椎側彎")
    ]

    for title, cx, val, label in views:
        # 檢測角度名稱
        draw.text((cx - 45, 60), title, fill="#94A3B8", font=font_sub)
        
        # 骨架人體圖
        # 頭部
        draw.ellipse([cx - 18, 90, cx + 18, 126], outline="#E2E8F0", width=2)
        # 脊椎中心線
        draw.line([cx, 126, cx, 340], fill="#64748B", width=2)
        # 肩膀
        draw.line([cx - 38, 155, cx + 38, 155], fill="#38BDF8", width=3)
        # 骨盆
        draw.line([cx - 32, 280, cx + 32, 280], fill="#38BDF8", width=3)
        # 下肢
        draw.line([cx - 18, 280, cx - 22, 450], fill="#94A3B8", width=2)
        draw.line([cx + 18, 280, cx + 22, 450], fill="#94A3B8", width=2)

        # 關節關鍵點
        draw.ellipse([cx - 4, 151, cx + 4, 159], fill="#38BDF8")
        draw.ellipse([cx - 4, 276, cx + 4, 284], fill="#38BDF8")

        # 數據標籤框
        color = "#EF4444" if val > 3.0 else "#10B981"
        draw.rectangle([cx - 45, 480, cx + 45, 525], fill="#1E293B", outline=color, width=2)
        
        # 繪製文字（修正水平置中）
        draw.text((cx - 28, 485), label, fill="#94A3B8", font=font_sub)
        draw.text((cx - 18, 503), f"{val:.1f}°", fill=color, font=font_val)

    # 底部說明文字
    draw.line([(20, 630), (width - 20, 630)], fill="#334155", width=1)
    draw.text((20, 642), "* 綠色代表中立位正常，紅色代表偏移過大需評估改善。", fill="#64748B", font=font_sub)
    
    return img

def draw_joint_icon(joint_name):
    """關節細節圖解 Icon (58x78)"""
    img = Image.new("RGB", (58, 78), "#F8FAFC")
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 57, 77], outline="#CBD5E1", width=1)
    
    draw.ellipse([19, 8, 39, 28], fill="#3B82F6")
    draw.line([29, 28, 29, 65], fill="#64748B", width=4)
    draw.ellipse([20, 52, 38, 70], fill="#0284C7")
    return img

def draw_muscle_pair(joint_name):
    """肌肉強弱對比圖 (140x118)"""
    img = Image.new("RGB", (140, 118), "#FFFFFF")
    draw = ImageDraw.Draw(img)
    font_bold = get_font(10, bold=True)
    font_small = get_font(9)

    draw.rectangle([0, 0, 139, 117], outline="#CBD5E1", width=1)

    # 緊繃肌肉
    draw.rounded_rectangle([6, 10, 134, 52], radius=4, fill="#FFE4E6", outline="#E11D48")
    draw.text((12, 14), "過度緊繃 / 縮短", fill="#9F1239", font=font_bold)
    draw.text((12, 30), "• 胸大肌 / 髂腰肌", fill="#E11D48", font=font_small)

    # 弱化肌肉
    draw.rounded_rectangle([6, 62, 134, 104], radius=4, fill="#E0F2FE", outline="#0284C7")
    draw.text((12, 66), "弱化無力 / 拉長", fill="#0369A1", font=font_bold)
    draw.text((12, 82), "• 中下斜方肌 / 菱形肌", fill="#0284C7", font=font_small)

    return img