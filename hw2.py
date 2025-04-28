import os
import pickle
import numpy as np
import pandas as pd

from scipy.signal import butter, lfilter
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import classification_report, accuracy_score

# === 匯入各種分類器 ===
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# ------------------------------------------------------------
# 1. 輔助函式：讀取單一受試者 (如 S2) 的 WESAD 資料
# ------------------------------------------------------------
def load_wesad_subject_data(subject_id, base_path):
    """
    傳回: (ecg, eda, emg, resp, temp, acc_x, acc_y, acc_z, labels)
    注意: 這裡示範載入 chest 資料
    """
    subject_folder = os.path.join(base_path, f'S{subject_id}')
    pkl_file = os.path.join(subject_folder, f'S{subject_id}.pkl')
    
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    
    labels = data[b'label']
    chest = data[b'signal'][b'chest']
    ecg = chest[b'ECG'][:, 0]
    eda = chest[b'EDA'][:, 0]
    emg = chest[b'EMG'][:, 0]
    resp = chest[b'Resp'][:, 0]
    temp = chest[b'Temp'][:, 0]
    acc_x = chest[b'ACC'][:, 0]
    acc_y = chest[b'ACC'][:, 1]
    acc_z = chest[b'ACC'][:, 2]
    
    return ecg, eda, emg, resp, temp, acc_x, acc_y, acc_z, labels


# ------------------------------------------------------------
# 2. 主程式開始：讀取多位受試者並合併
# ------------------------------------------------------------
base_path = r".\WESAD"  # <-- 請自行改成實際的 WESAD 路徑
subject_ids = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]   # 這裡示範一次合併 S2, S3, S4，視資料情況可再擴增
LABELS_TO_USE = [1, 2]    # 僅保留標籤 1 (中性) 與 2 (壓力)

# 用 list 暫存
all_ecg, all_eda, all_emg = [], [], []
all_resp, all_temp = [], []
all_accx, all_accy, all_accz = [], [], []
all_labels = []
all_subjects = []   # 每筆資料的 subject ID

for sid in subject_ids:
    ecg, eda, emg, resp, temp, ax, ay, az, labels = load_wesad_subject_data(sid, base_path)

    # 串接每個受試者的訊號
    all_ecg.append(ecg)
    all_eda.append(eda)
    all_emg.append(emg)
    all_resp.append(resp)
    all_temp.append(temp)
    all_accx.append(ax)
    all_accy.append(ay)
    all_accz.append(az)
    all_labels.append(labels)
    
    # 每筆資料都標記它來自 sid
    all_subjects.append( np.full(shape=len(labels), fill_value=sid) )

# 把多位受試者資料串接成單一 numpy array
ecg = np.concatenate(all_ecg)
eda = np.concatenate(all_eda)
emg = np.concatenate(all_emg)
resp = np.concatenate(all_resp)
temp = np.concatenate(all_temp)
acc_x = np.concatenate(all_accx)
acc_y = np.concatenate(all_accy)
acc_z = np.concatenate(all_accz)
labels = np.concatenate(all_labels)
subjects = np.concatenate(all_subjects)

# ------------------------------------------------------------
# 3. 篩選目標標籤 (1,2)，並移除其他標籤
# ------------------------------------------------------------
valid_idx = np.isin(labels, LABELS_TO_USE)
ecg = ecg[valid_idx]
eda = eda[valid_idx]
emg = emg[valid_idx]
resp = resp[valid_idx]
temp = temp[valid_idx]
acc_x = acc_x[valid_idx]
acc_y = acc_y[valid_idx]
acc_z = acc_z[valid_idx]
labels = labels[valid_idx]
subjects = subjects[valid_idx]

# ------------------------------------------------------------
# 4. 視窗切分（同時保留 subjects）
# ------------------------------------------------------------
WINDOW_SIZE = 700 * 3  # 700Hz x 3秒

def segment_data_and_subjects(*signals, labels, subjects, window_size=WINDOW_SIZE):
    """
    signals: 例如 ecg, eda, ...
    labels:  單一路徑的 label (array)
    subjects: 每筆資料的 subject ID
    window_size: 視窗大小 (樣本數)
    回傳: (list_of_windows, list_of_labels, list_of_subjects)
    """
    min_len = min([len(s) for s in signals] + [len(labels)] + [len(subjects)])
    n_segments = min_len // window_size
    
    seg_data = []
    seg_labels = []
    seg_subjs = []
    
    for i in range(n_segments):
        start = i * window_size
        end = start + window_size
        
        # 取出每個信號在此視窗的切片
        window_sigs = [s[start:end] for s in signals]
        # 決定此視窗的標籤與 subject：以視窗最後一筆資料為準 (可改成最常出現的標籤…)
        window_label = labels[end - 1]
        window_subj  = subjects[end - 1]
        
        seg_data.append(window_sigs)
        seg_labels.append(window_label)
        seg_subjs.append(window_subj)
    
    return seg_data, seg_labels, seg_subjs

seg_data, seg_labels, seg_subjects = segment_data_and_subjects(
    ecg, eda, emg, resp, temp, acc_x, acc_y, acc_z,
    labels=labels,
    subjects=subjects,
    window_size=WINDOW_SIZE
)

# ------------------------------------------------------------
# 5. 特徵擷取 (包含時域 & 頻域)
# ------------------------------------------------------------
def butter_bandpass_filter(signal, lowcut, highcut, fs, order=5):
    b, a = butter(order, [lowcut/(0.5*fs), highcut/(0.5*fs)], btype='band')
    return lfilter(b, a, signal)

def compute_band_power(signal, fs, band):
    """
    計算 signal 在指定頻帶 (band=(low, high)) 內的能量(積分)。
    可用濾波 -> 能量 或直接用 Welch PSD 再積分。
    """
    filtered = butter_bandpass_filter(signal, band[0], band[1], fs=fs, order=4)
    return np.sum(filtered**2)

def extract_features_from_window(window_sigs, fs=700):
    """
    針對單一視窗內的多通道訊號，回傳一組特徵。
    window_sigs = [ecg_window, eda_window, emg_window, resp_window, temp_window, accx_window, accy_window, accz_window]
    """
    ecg_w, eda_w, emg_w, resp_w, temp_w, accx_w, accy_w, accz_w = window_sigs
    
    feats = {}
    
    # (A) 時域特徵: mean, std, min, max
    def basic_stats(sig, prefix):
        feats[f'{prefix}_mean'] = np.mean(sig)
        feats[f'{prefix}_std']  = np.std(sig)
        feats[f'{prefix}_min']  = np.min(sig)
        feats[f'{prefix}_max']  = np.max(sig)
    
    basic_stats(ecg_w,  'ecg')
    basic_stats(eda_w,  'eda')
    basic_stats(emg_w,  'emg')
    basic_stats(resp_w, 'resp')
    basic_stats(temp_w, 'temp')
    basic_stats(accx_w, 'accx')
    basic_stats(accy_w, 'accy')
    basic_stats(accz_w, 'accz')
    
    # (B) 頻域特徵: 以 ECG 為例計算 ULF, LF, HF, UHF 能量
    freq_bands = {
        'ULF': (0.01, 0.04),
        'LF':  (0.04, 0.15),
        'HF':  (0.15, 0.40),
        'UHF': (0.40, 1.00),
    }
    for name, (low, high) in freq_bands.items():
        band_power = compute_band_power(ecg_w, fs, (low, high))
        feats[f'ecg_{name}_power'] = band_power
    
    return feats

all_features = []
for w_sigs in seg_data:
    feats = extract_features_from_window(w_sigs, fs=700)
    all_features.append(feats)

feature_df = pd.DataFrame(all_features)
label_array = np.array(seg_labels)
subject_array = np.array(seg_subjects)

# ------------------------------------------------------------
# 6. 使用 LOSO 驗證 + 多模型比較
# ------------------------------------------------------------
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "AdaBoost": AdaBoostClassifier(
        estimator=DecisionTreeClassifier(), n_estimators=50, random_state=42
    ),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "LDA": LinearDiscriminantAnalysis(),
    "DecisionTree": DecisionTreeClassifier(random_state=42)
}

logo = LeaveOneGroupOut() 

for model_name, clf in models.items():
    # 使用 cross_val_predict 搭配 LOSO
    y_pred = cross_val_predict(
        clf, feature_df, label_array,
        cv=logo,
        groups=subject_array  # 以 subject_array 為分組依據 (LOSO)
    )
    acc = accuracy_score(label_array, y_pred)
    print(f"\n[{model_name}] LOSO Accuracy: {acc:.4f}")
    report = classification_report(label_array, y_pred, digits=4)
    print(report)
