import numpy as np

def calculate_shoulder_rom_and_compensation(landmarks: dict, side: str = 'R') -> dict:
    """
    計算扣除軀幹代償後的肩關節屈曲、外展角度以及聳肩代償指標。

    參數:
        landmarks (dict): 包含 3D 關節座標的字典，數值為 np.array([x, y, z])。
                         需要包含: 'L_Shoulder', 'R_Shoulder', 'L_Hip', 'R_Hip', 'Neck', 'L_Elbow', 'R_Elbow'
        side (str): 'R' 代表右肩，'L' 代表左肩。預設為 'R'。

    傳回:
        dict: 包含真實屈曲角度、外展角度、軀幹代償角度與聳肩指標的字典。
    """
    # -------------------------------------------------------------------------
    # 1. 讀取並轉換關鍵關節點座標 (確保為 NumPy 陣列)
    # -------------------------------------------------------------------------
    try:
        p_l_shoulder = np.array(landmarks['L_Shoulder'], dtype=np.float64)
        p_r_shoulder = np.array(landmarks['R_Shoulder'], dtype=np.float64)
        p_l_hip = np.array(landmarks['L_Hip'], dtype=np.float64)
        p_r_hip = np.array(landmarks['R_Hip'], dtype=np.float64)
        
        # 若無直接提供 Neck，可用雙肩中點替代
        if 'Neck' in landmarks and landmarks['Neck'] is not None:
            p_neck = np.array(landmarks['Neck'], dtype=np.float64)
        else:
            p_neck = (p_l_shoulder + p_r_shoulder) / 2.0
            
        p_elbow = np.array(landmarks[f'{side}_Elbow'], dtype=np.float64)
        p_target_shoulder = p_r_shoulder if side == 'R' else p_l_shoulder
    except KeyError as e:
        raise KeyError(f"缺少必要的關節座標點: {e}")

    p_mid_hip = (p_l_hip + p_r_hip) / 2.0  # 骨盆中心點

    # -------------------------------------------------------------------------
    # 2. 建立軀幹局部解剖坐標系 (Local Trunk Frame)
    # -------------------------------------------------------------------------
    # Y 軸 (軀幹長軸)：由骨盆中心指向頸部/胸骨
    v_spine = p_neck - p_mid_hip
    spine_length = np.linalg.norm(v_spine)
    if spine_length == 0:
        raise ValueError("軀幹長度為 0，請檢查關節點座標。")
    y_trunk = v_spine / spine_length  # 歸一化 Y 軸向量

    # 橫軸 (雙肩軸)：右肩指向左肩
    v_shoulder_line = p_l_shoulder - p_r_shoulder
    
    # Z 軸 (軀幹前向軸)：透過外積 (Transverse × Spine) 計算垂直於胸壁向前的向量
    v_front = np.cross(v_shoulder_line, v_spine)
    front_length = np.linalg.norm(v_front)
    if front_length == 0:
        raise ValueError("無法計算軀幹前向向量，請檢查關節點是否共線。")
    z_trunk = v_front / front_length  # 歸一化 Z 軸向量

    # X 軸 (校正後的軀幹橫軸)：確保三軸相互正交 (y_trunk × z_trunk)
    x_trunk = np.cross(y_trunk, z_trunk)
    x_trunk = x_trunk / np.linalg.norm(x_trunk)  # 歸一化 X 軸向量

    # -------------------------------------------------------------------------
    # 3. 計算上臂向量與方向投影
    # -------------------------------------------------------------------------
    # 上臂向量：肩膀指向肘關節
    v_arm = p_elbow - p_target_shoulder
    arm_length = np.linalg.norm(v_arm)
    if arm_length == 0:
        raise ValueError("上臂長度為 0，請檢查肘關節座標。")
    a_unit = v_arm / arm_length  # 單位上臂向量

    # 軀幹向下向量 (-Y)
    y_trunk_down = -y_trunk

    # --- 角度 A: 肩關節外展 (Abduction) ---
    # 投影至冠狀面 (Coronal Plane: 由 X 軸與 Y 軸張成)
    a_coronal = np.dot(a_unit, x_trunk) * x_trunk + np.dot(a_unit, y_trunk) * y_trunk
    a_coronal_norm = np.linalg.norm(a_coronal)
    
    if a_coronal_norm > 1e-6:
        cos_abd = np.dot(a_coronal, y_trunk_down) / a_coronal_norm
        cos_abd = np.clip(cos_abd, -1.0, 1.0)  # 防止浮點數誤差導致超出 [-1, 1]
        abd_angle = np.degrees(np.arccos(cos_abd))
    else:
        abd_angle = 0.0

    # --- 角度 B: 肩關節屈曲 (Flexion) ---
    # 投影至矢狀面 (Sagittal Plane: 由 Z 軸與 Y 軸張成)
    a_sagittal = np.dot(a_unit, z_trunk) * z_trunk + np.dot(a_unit, y_trunk) * y_trunk
    a_sagittal_norm = np.linalg.norm(a_sagittal)
    
    if a_sagittal_norm > 1e-6:
        cos_flex = np.dot(a_sagittal, y_trunk_down) / a_sagittal_norm
        cos_flex = np.clip(cos_flex, -1.0, 1.0)
        flex_angle = np.degrees(np.arccos(cos_flex))
    else:
        flex_angle = 0.0

    # -------------------------------------------------------------------------
    # 4. 代償指標計算
    # -------------------------------------------------------------------------
    # A. 軀幹前後傾角 (Trunk Pitch / Flexion-Extension Compensation)
    # 計算 Y_trunk 相對於全域垂直軸 (Global Z 或 Y) 的傾角 (假設全域 Y 為向上)
    global_up = np.array([0.0, 1.0, 0.0])  # 若採集系統全域垂直為 Z 軸，請修改為 [0, 0, 1]
    cos_trunk_tilt = np.dot(y_trunk, global_up) / (np.linalg.norm(y_trunk) * np.linalg.norm(global_up))
    trunk_tilt_angle = np.degrees(np.arccos(np.clip(cos_trunk_tilt, -1.0, 1.0)))

    # B. 聳肩代償指標 (Scapular Elevation Metric)
    # 計算目標肩膀相對於 Neck 在軀幹長軸 (y_trunk) 上的相對高度位移，並以軀幹長度歸一化
    v_neck_to_shoulder = p_target_shoulder - p_neck
    shoulder_elevation_height = np.dot(v_neck_to_shoulder, y_trunk)
    shrug_index = shoulder_elevation_height / spine_length

    # -------------------------------------------------------------------------
    # 5. 輸出結果打包
    # -------------------------------------------------------------------------
    return {
        "side": side,
        "clean_abduction_deg": round(float(abd_angle), 2),     # 純肩外展角度 (已扣除側彎)
        "clean_flexion_deg": round(float(flex_angle), 2),       # 純肩屈曲角度 (已扣除後仰)
        "trunk_tilt_compensation_deg": round(float(trunk_tilt_angle), 2), # 軀幹傾斜代償角度
        "shrug_index": round(float(shrug_index), 4),            # 聳肩指數 (數值越正代表聳肩越明顯)
        "is_shrugging_warning": shrug_index > 0.05              # 可自訂閾值 (例如 > 0.05 觸發警告)
    }

# =============================================================================
# 測試範例 (Test Run)
# =============================================================================
if __name__ == "__main__":
    # 模擬 3D 關節座標資料 (單位: 公尺/毫米皆可)
    mock_landmarks = {
        'L_Shoulder': np.array([0.20,  1.45, 0.00]),
        'R_Shoulder': np.array([-0.20, 1.50, 0.00]), # 右肩有些許聳起
        'L_Hip':      np.array([0.15,  0.90, 0.00]),
        'R_Hip':      np.array([-0.15, 0.90, 0.00]),
        'Neck':       np.array([0.00,  1.48, 0.00]),
        'R_Elbow':    np.array([-0.20, 1.85, 0.35])  # 右臂抬起 (向前+向上)
    }

    result = calculate_shoulder_rom_and_compensation(mock_landmarks, side='R')
    
    print("=== 肩關節 ROM 與代償評估結果 ===")
    for key, val in result.items():
        print(f"{key}: {val}")