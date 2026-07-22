import gc
import cv2
import numpy as np
import streamlit as st
from deepface import DeepFace

st.set_page_config(
    page_title="AI 人臉分析系統",
    page_icon="👤"
)

st.title("👤 AI 人臉分析系統")

uploaded_file = st.file_uploader(
    "上傳照片",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    # 讀取圖片 byte 並轉換為 numpy array
    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # OpenCV 預設 BGR -> 轉成 RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    st.image(img_rgb, caption="上傳圖片", use_column_width=True)

    if st.button("開始分析"):
        with st.spinner("AI 分析中..."):
            try:
                # 傳入 img_rgb 可以節省內部轉碼開銷
                result = DeepFace.analyze(
                    img_path=img_rgb,
                    actions=[
                        "age",
                        "gender",
                        "emotion",
                        "race"  # ⚠️ 若在免費版 Streamlit Cloud 依然重啟，可嘗試把 "race" 註解掉
                    ],
                    detector_backend="mediapipe",
                    enforce_detection=False
                )

                # 處理 DeepFace 回傳的格式 (List 或 Dict)
                res = result[0] if isinstance(result, list) else result

                st.success("分析完成！")

                # 排版顯示結果
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("🎂 年齡", f"{res.get('age')} 歲")
                    st.metric("👤 性別", res.get("dominant_gender", "未知"))
                with col2:
                    st.metric("😊 情緒", res.get("dominant_emotion", "未知"))
                    st.metric("🌏 族群", res.get("dominant_race", "未知"))

            except Exception as e:
                st.error(f"分析失敗：{e}")

            finally:
                # 核心關鍵：手動釋放 Python 及 Tensorflow 暫存記憶體
                gc.collect()
