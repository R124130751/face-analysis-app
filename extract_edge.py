import os
import cv2
import gradio as gr
import numpy as np

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


# =========================
# 鬼滅風格核心影像處理演算法
# =========================
def enhance_contrast_and_saturation(
    img: np.ndarray, saturation_scale: float = 1.8, contrast_scale: float = 1.3
) -> np.ndarray:
    """提升色彩鮮豔度與對比度（營造鬼滅高強度的光影與色彩）"""
    # 1. 對比度與亮度調整 (CLAHE 局部對比增強)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    img_contrast = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # 2. 飽和度增強
    hsv = cv2.cvtColor(img_contrast, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_scale, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def smooth_color_bilateral(img: np.ndarray, times: int = 7) -> np.ndarray:
    """多重雙邊濾波：讓肌膚與服裝呈現象徵動漫的繪畫色塊感"""
    result = img.copy()
    for _ in range(int(times)):
        result = cv2.bilateralFilter(result, d=9, sigmaColor=85, sigmaSpace=85)
    return result


def extract_kimetsu_ink_lines(
    img: np.ndarray, line_thickness: int = 2, sensitivity: int = 3
) -> np.ndarray:
    """擷取鬼滅風格靈魂：濃密強烈的水墨黑粗線條"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 5)

    # 自適應閾值提線
    block_size = 9
    c = sensitivity
    edges = cv2.adaptiveThreshold(
        gray_blur,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c,
    )

    # 粗線化處理 (侵蝕黑線/膨脹遮罩，強化浮世繪筆觸)
    if line_thickness > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (line_thickness, line_thickness)
        )
        edges = cv2.erode(edges, kernel)

    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


def convert_to_kimetsu_style(
    input_img: np.ndarray,
    saturation: float,
    bilateral_times: int,
    line_thickness: int,
    line_sensitivity: int,
) -> np.ndarray:
    """風格轉換主程式"""
    if input_img is None:
        return None

    # Gradio RGB 轉 OpenCV BGR
    img_bgr = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)

    # 1. 色彩與對比度拉升 (鮮豔光感)
    img_vibrant = enhance_contrast_and_saturation(
        img_bgr, saturation_scale=saturation
    )

    # 2. 賽璐珞色塊平滑 (繪圖感)
    img_smooth = smooth_color_bilateral(img_vibrant, times=bilateral_times)

    # 3. 鬼滅水墨黑線繪製
    img_lines = extract_kimetsu_ink_lines(
        img_bgr, line_thickness=line_thickness, sensitivity=line_sensitivity
    )

    # 4. 疊合黑線與色塊 (Bitwise AND)
    img_anime_bgr = cv2.bitwise_and(img_smooth, img_lines)

    # 轉回 RGB 供 Web UI 顯示
    return cv2.cvtColor(img_anime_bgr, cv2.COLOR_BGR2RGB)


# =========================
# Gradio 網頁介面佈局
# =========================
with gr.Blocks(title="鬼滅水墨動漫風格轉換器") as demo:
    gr.Markdown(
        """
        # ⚔️ 鬼滅之刃 (Demon Slayer) 水墨動漫風格轉換器
        上傳照片後，可一鍵套用**鬼滅風格預設**，或調整參數控制濃重黑線與鮮豔色彩！
        """
    )

    with gr.Row():
        # 左側控制器
        with gr.Column(scale=1):
            input_image = gr.Image(label="1. 上傳原圖", type="numpy")

            # 快速套用預設按鈕
            btn_kimetsu_preset = gr.Button(
                "🔥 一鍵套用【鬼滅強烈水墨風】預設", variant="secondary"
            )

            gr.Markdown("### 🎛️ 微調控制項")
            saturation_slider = gr.Slider(
                minimum=1.0,
                maximum=3.0,
                value=2.0,
                step=0.1,
                label="色彩鮮豔度/對比 (Saturation & Bright)",
            )
            times_slider = gr.Slider(
                minimum=1,
                maximum=12,
                value=8,
                step=1,
                label="賽璐珞色塊平滑度 (Smoothness)",
            )
            thickness_slider = gr.Slider(
                minimum=1,
                maximum=5,
                value=2,
                step=1,
                label="黑線粗細 (Line Thickness - 鬼滅特色選 2~3)",
            )
            sensitivity_slider = gr.Slider(
                minimum=1,
                maximum=8,
                value=3,
                step=1,
                label="線條豐富度 (數字越小線條越濃密)",
            )

            btn_convert = gr.Button("🚀 開始轉換", variant="primary")

        # 右側結果展示
        with gr.Column(scale=1):
            output_image = gr.Image(label="2. 鬼滅風格轉換結果")

    # --- 事件連結 ---
    # 點擊開始轉換
    btn_convert.click(
        fn=convert_to_kimetsu_style,
        inputs=[
            input_image,
            saturation_slider,
            times_slider,
            thickness_slider,
            sensitivity_slider,
        ],
        outputs=output_image,
    )

    # 鬼滅預設按鈕：重置參數並自動觸發轉換
    def apply_preset(img):
        if img is None:
            return None, 2.0, 8, 2, 3
        res = convert_to_kimetsu_style(img, 2.0, 8, 2, 3)
        return res, 2.0, 8, 2, 3

    btn_kimetsu_preset.click(
        fn=apply_preset,
        inputs=[input_image],
        outputs=[
            output_image,
            saturation_slider,
            times_slider,
            thickness_slider,
            sensitivity_slider,
        ],
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)