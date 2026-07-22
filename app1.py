import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace


st.title("👤 AI 人臉分析系統")


uploaded_file = st.file_uploader(
    "上傳照片",
    type=["jpg","png","jpeg"]
)


if uploaded_file:

    bytes_data = uploaded_file.read()


    img = np.frombuffer(
        bytes_data,
        np.uint8
    )


    img = cv2.imdecode(
        img,
        cv2.IMREAD_COLOR
    )


    st.image(
        img,
        channels="BGR"
    )


    if st.button("開始分析"):


        with st.spinner(
            "AI分析中..."
        ):


            result = DeepFace.analyze(
                img_path=img,
                actions=[
                    "age",
                    "gender",
                    "emotion"
                ],
                detector_backend="retinaface",
                enforce_detection=False
            )


        if isinstance(result,list):
            res=result[0]
        else:
            res=result


        st.success(
            "分析完成"
        )


        st.write(
            "👶 年齡:",
            res.get("age")
        )


        st.write(
            "🚻 性別:",
            res.get(
                "dominant_gender",
                res.get("gender")
            )
        )


        st.write(
            "😀 情緒:",
            res.get(
                "dominant_emotion"
            )
        )

        st.write(
            "情緒:",
            res["dominant_emotion"]
        )
