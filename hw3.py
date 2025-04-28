import os
import pickle
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adam

base_path = os.path.join(os.path.dirname(__file__), 'WESAD')
subject_ids = [f"S{i}" for i in range(2, 17) if i not in [3, 12]]

all_data = []

for sid in subject_ids:
    pickle_file = os.path.join(base_path, sid, f"{sid}.pkl")
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f, encoding='bytes')

    label = data[b'label']
    chest_data = data[b'signal'][b'chest']

    df_dict = {
        "ACC":  chest_data[b'ACC'][:, 0],
        "ECG":  chest_data[b'ECG'][:, 0],
        "EMG":  chest_data[b'EMG'][:, 0],
        "EDA":  chest_data[b'EDA'][:, 0],
        "Resp": chest_data[b'Resp'][:, 0],
        "Temp": chest_data[b'Temp'][:, 0],
        "label": label
    }
    df = pd.DataFrame(df_dict)
    df["subject"] = sid
    all_data.append(df)

# 合併所有 subject 的資料
df_all = pd.concat(all_data, ignore_index=True)

# 只保留 label=1 (中性) 與 label=2 (壓力)
df_all = df_all[df_all['label'].isin([1, 2])]

df_sampled = (
    df_all.groupby(['subject', 'label'])
    .apply(lambda x: x.sample(n=40, random_state=42))
    .reset_index(drop=True)
)

X = df_sampled.drop(columns=['label', 'subject']).values  # shape = (num_samples, 6)
y = df_sampled['label'].values                            # shape = (num_samples,)
y = y - 1  # label 1,2 -> 0,1

# train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

X_train_3d = X_train.reshape((X_train.shape[0], 6, 1))
X_test_3d  = X_test.reshape((X_test.shape[0], 6, 1))

lstm_model = Sequential()
lstm_model.add(LSTM(64, return_sequences=False, input_shape=(6, 1)))
lstm_model.add(BatchNormalization())
lstm_model.add(Flatten())
lstm_model.add(Dense(64, activation='relu'))
lstm_model.add(Dropout(0.5))
lstm_model.add(Dense(32, activation='relu'))
lstm_model.add(Dense(1, activation='sigmoid')) 

lstm_model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

lstm_model.fit(X_train_3d, y_train, epochs=50, batch_size=16, validation_split=0.2, verbose=1)

y_pred_proba = lstm_model.predict(X_test_3d)
y_pred = (y_pred_proba > 0.5).astype(int).ravel()

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=['class 0', 'class 1']))
    