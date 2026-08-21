import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation

cap = cv2.VideoCapture(0)
bg = cv2.imread('windows-bg.jpg')

# 檢查背景圖是否存在
if bg is None:
    print("無法載入背景圖片 windows-bg.jpg，請確認檔案路徑！")
    exit()

# 預先將背景圖調整為與視訊畫面相同的 520x300 尺寸
bg = cv2.resize(bg, (520, 300))

with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
    if not cap.isOpened():
        print("無法開啟攝影機")
        exit()

    while True:
        ret, img = cap.read()
        if not ret:
            print("無法取得攝影機畫面")
            break

        img = cv2.resize(img, (520, 300))
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        results = selfie_segmentation.process(img2)
        
        # 建立 3 通道的 True/False 遮罩
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
        
        # 滿足 condition (>0.1) 的像素用原圖 (img)，其餘用背景圖 (bg)
        output_image = np.where(condition, img, bg)

        cv2.imshow('Virtual Background', output_image)
        if cv2.waitKey(5) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()