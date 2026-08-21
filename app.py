import os
import base64
import random  # 👈 引入隨機模組
import cv2
import numpy as np
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# 1. 載入模型與辨識器
recog = cv2.face.LBPHFaceRecognizer_create()
if os.path.exists('face.yml'):
    recog.read('face.yml')

detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# 2. 標籤對應表與關鍵字對映
label_names = {1: 'Messi', 2: 'Cristiano Ronaldo'}

# 支持多種輸入關鍵字的映射
target_map = {
    'Messiesi': 'face01',
    'Messi': 'face01',
    '梅西': 'face01',
    'C羅': 'face02',
    'C羅納度': 'face02',
    'ronaldo': 'face02',
    'cr7': 'face02'
}

# 3. HTML 前端頁面 (改成文字輸入框)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <title>AI 球星人臉辨識盲盒</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; text-align: center; background: #eef2f5; padding: 40px; }
        .card { background: white; max-width: 550px; margin: 0 auto; padding: 30px; border-radius: 16px; box-shadow: 0 8px 20px rgba(0,0,0,0.1); }
        input[type="text"] { font-size: 16px; padding: 10px 15px; width: 60%; border-radius: 8px; border: 2px solid #ddd; outline: none; transition: 0.3s; }
        input[type="text"]:focus { border-color: #007bff; }
        button { font-size: 16px; padding: 10px 20px; margin-left: 8px; border-radius: 8px; border: none; background: #007bff; color: white; cursor: pointer; transition: 0.2s; }
        button:hover { background: #0056b3; }
        #result-container { margin-top: 25px; }
        #result-img { max-width: 100%; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.15); display: none; }
        #loading { display: none; color: #666; font-weight: bold; margin-top: 15px; }
        .hint { color: #888; font-size: 14px; margin-top: 8px; }
    </style>
</head>
<body>
    <div class="card">
        <h2>⚽ 球星 AI 人臉辨識盲盒</h2>
        <p>請輸入你想尋找的球星名稱：</p>
        <div>
            <input type="text" id="person-input" placeholder="例如：mesi, c羅, 梅西" onkeydown="if(event.key==='Enter') predictFace()">
            <button onclick="predictFace()">抽樣辨識</button>
        </div>
        <div class="hint">💡 提示：每次搜尋都會從該球星的照片庫隨機抽出一張！</div>
        
        <div id="loading">🔍 正在隨機抽取照片並進行 AI 辨識...</div>
        <div id="result-container">
            <img id="result-img" src="" alt="辨識結果">
        </div>
    </div>

    <script>
        async function predictFace() {
            const inputVal = document.getElementById('person-input').value.trim();
            const loading = document.getElementById('loading');
            const img = document.getElementById('result-img');

            if (!inputVal) {
                alert('請先輸入名稱！');
                return;
            }
            
            loading.style.display = 'block';
            img.style.display = 'none';

            try {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ person: inputVal })
                });

                const data = await response.json();
                loading.style.display = 'none';

                if (data.status === 'success') {
                    img.src = 'data:image/jpeg;base64,' + data.image;
                    img.style.display = 'inline-block';
                } else {
                    alert('❌ ' + data.message);
                }
            } catch (err) {
                loading.style.display = 'none';
                alert('連線失敗，請檢查伺服器狀態。');
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    person_key = data.get('person', '').strip().lower()

    # 1. 檢查輸入關鍵字
    if person_key not in target_map:
        return jsonify({
            'status': 'error', 
            'message': f"找不到 '{person_key}'！目前支援：mesi, messi, 梅西, c羅, ronaldo, cr7"
        })

    folder = target_map[person_key]
    if not os.path.exists(folder):
        return jsonify({'status': 'error', 'message': f'伺服器找不到資料夾 {folder}'})

    # 2. 抓取資料夾內所有圖片
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
    if not files:
        return jsonify({'status': 'error', 'message': f'資料夾 {folder} 內沒有圖片照片'})

    # 3. 🎲 隨機選取一張圖片！(每次請求都會變)
    selected_file = random.choice(files)
    img_path = os.path.join(folder, selected_file)
# 在 Linux 下 imdecode + np.fromfile 同樣能完美運作
img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    # 4. 讀取照片並進行人臉辨識
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        val, confidence = recog.predict(face_roi)
        
        name = label_names.get(val, "Unknown")
        # LBPH confidence 數值越低越匹配
        text = f"{name} ({round(confidence, 1)})" if confidence < 80 else "Unknown"

        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 5. 編碼轉成 Base64 傳回前端頁面
    _, buffer = cv2.imencode('.jpg', img)
    img_as_text = base64.b64encode(buffer).decode('utf-8')

    return jsonify({'status': 'success', 'image': img_as_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)