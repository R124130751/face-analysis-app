import os
import cv2
import numpy as np

def load_image_with_unicode(file_path):
    """ 安全讀取包含中文或特殊字元路徑的圖片 """
    try:
        img_array = np.fromfile(file_path, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        return None

def search_and_show_face():
    faces_dir = "faces"

    # 1. 檢查 faces 資料夾是否存在
    if not os.path.exists(faces_dir):
        print(f"❌ 錯誤：找不到資料夾 '{faces_dir}'，請先建立該資料夾並放入圖片！")
        return

    # 2. 讓使用者輸入姓名
    name = input("請輸入要查詢的人員姓名 (例如: 張三 或 Alex)：").strip()

    if not name:
        print("⚠️ 未輸入姓名，程式結束。")
        return

    # 3. 搜尋匹配的圖片檔
    supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    target_path = None
    
    # 遍歷資料夾中的所有檔案
    for file_name in os.listdir(faces_dir):
        file_base, ext = os.path.splitext(file_name)
        # 不區分大小寫匹配檔名，且副檔名需為圖片格式
        if file_base.lower() == name.lower() and ext.lower() in supported_extensions:
            target_path = os.path.join(faces_dir, file_name)
            break

    # 4. 如果找到了照片就顯示，否則提示找不到
    if target_path:
        print(f"🔍 找到照片：{target_path}")
        img = load_image_with_unicode(target_path)
        
        if img is not None:
            # 顯示圖片
            cv2.imshow(f"Face Result: {name}", img)
            print("按任意鍵關閉圖片視窗...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("❌ 圖片檔案損毀或格式不支援，無法開啟。")
    else:
        print(f"❌ 查無此人！在 '{faces_dir}' 資料夾中找不到名為 '{name}' 的照片。")

if __name__ == "__main__":
    search_and_show_face()