import streamlit as st
import cv2
import numpy as np
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

    # 讀取圖片
    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )


    img = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )


    # OpenCV BGR -> RGB
    img_rgb = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )


    st.image(
        img_rgb,
        caption="上傳圖片"
    )


    if st.button("開始分析"):

        with st.spinner("AI 分析中..."):

            try:

                result = DeepFace.analyze(
                    img_path=img,
                    actions=[
                        "age",
                        "gender",
                        "emotion",
                        "race"
                    ],
                    detector_backend="mediapipe",
                    enforce_detection=False
                )


                # DeepFace新版處理
                if isinstance(result, list):
                    res = result[0]
                else:
                    res = result


                st.success("分析完成")


                st.write(
                    "🎂 年齡:",
                    res.get("age")
                )


                st.write(
                    "👤 性別:",
                    res.get("dominant_gender")
                )


                st.write(
                    "😊 情緒:",
                    res.get("dominant_emotion")
                )


                st.write(
                    "🌏 族群:",
                    res.get("dominant_race")
                )


            except Exception as e:

                st.error(
                    f"分析失敗：{e}"
                )
