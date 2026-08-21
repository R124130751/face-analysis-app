import cv2

# 1. 讀取圖片
img = cv2.imread('faces/852.jpg')

# 防護機制：檢查圖片是否成功讀取
if img is None:
    print("❌ 錯誤：找不到 '852.jpg'，請確認圖片檔案與 .py 檔在同一目錄下！")
else:
    # 2. 轉為灰階影像並載入人臉模型
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    # 3. 偵測人臉
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3)

    # 4. 針對每張偵測到的臉進行馬賽克處理
    for (x, y, w, h) in faces:
        mosaic = img[y:y+h, x:x+w]    # 切下人臉區域
        level = 15                   # 馬賽克程度（數值越大越模糊）

        # 確保縮小後的尺寸至少為 1x1，防止過小報錯
        mh = max(1, int(h / level))
        mw = max(1, int(w / level))

        # 先縮小（將細節丟棄），再放大（產生鋸齒方格感）
        mosaic = cv2.resize(mosaic, (mw, mh), interpolation=cv2.INTER_LINEAR)
        mosaic = cv2.resize(mosaic, (w, h), interpolation=cv2.INTER_NEAREST)

        # 蓋回原圖區域
        img[y:y+h, x:x+w] = mosaic

    # 5. 顯示成果
    cv2.imshow('oxxostudio', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()