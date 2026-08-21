import pyautogui
import time

# 貼心小提示：延遲 2 秒，讓你有時間切換到你想截圖的畫面
print("2 秒後開始截圖，請切換到目標畫面...")
time.sleep(2)

# 執行截圖
myScreenshot = pyautogui.screenshot()

# 儲存圖片（直接存放在跟這支程式碼同一個資料夾下，避免路徑寫錯報錯）
myScreenshot.save('my_screenshot.png')
print("截圖成功！圖片已儲存為 my_screenshot.png")