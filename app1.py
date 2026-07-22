import streamlit as st
from deepface import DeepFace
import cv2
import numpy as np


st.title("👤 AI 人臉分析系統")


uploaded_file = st.file_uploader(
    "上傳照片",
    type=["jpg","png","jpeg"]
)


if uploaded_file:

    img = np.array(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
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

        result = DeepFace.analyze(
            img,
            actions=[
                "age",
                "gender",
                "emotion",
                "race"
            ],
            detector_backend="mediapipe",
            enforce_detection=False,
        )


        res=result[0]


        st.write(
            "年齡:",
            res["age"]
        )

        st.write(
            "性別:",
            res["dominant_gender"]
        )

        st.write(
            "情緒:",
            res["dominant_emotion"]
        )
