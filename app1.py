import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from deepface import DeepFace
import gradio as gr


def analyze_face(image):
    if image is None:
        return "請先上傳照片！"

    try:
        results = DeepFace.analyze(
            img_path=image,
            actions=['age', 'gender', 'emotion', 'race'],
            detector_backend='mediapipe',
            enforce_detection=False,
        )

        res = results[0]

        return f"""
        ### 📊 分析結果：
        * **推定年齡**：{res.get('age', '未知')} 歲
        * **性別**：{res.get('dominant_gender', '未知')}
        * **主要情緒**：{res.get('dominant_emotion', '未知')}
        * **種族/族群**：{res.get('dominant_race', '未知')}
        """
    except Exception as e:
        return f"❌ 分析失敗：{str(e)}"


demo = gr.Interface(
    fn=analyze_face,
    inputs=gr.Image(type="numpy", label="上傳照片"),
    outputs=gr.Markdown(label="分析結果"),
    title="👤 AI 人臉分析系統",
)

if __name__ == "__main__":
    demo.launch()
