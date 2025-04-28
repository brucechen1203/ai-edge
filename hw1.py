import numpy as np
import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


base_path = os.path.join(os.path.dirname(__file__), 'WESAD')
subject_path = os.path.join(base_path, 'S2')
pickle_file = os.path.join(subject_path, 'S2.pkl')

with open(pickle_file, 'rb') as f:
    data = pickle.load(f, encoding='bytes')


label = data[b'label']  
chest_data = data[b'signal'][b'chest']

df_dict = {
    "ACC": chest_data[b'ACC'][:, 0],
    "ECG": chest_data[b'ECG'][:, 0],
    "EMG": chest_data[b'EMG'][:, 0],
    "EDA": chest_data[b'EDA'][:, 0],
    "Resp": chest_data[b'Resp'][:, 0],
    "Temp": chest_data[b'Temp'][:, 0],
    "label": label
}

# 轉換為 DataFrame
df = pd.DataFrame(df_dict)

# 中性 vs 壓力
df = df[df['label'].isin([1, 2])]

df_sampled = df.groupby('label').apply(lambda x: x.sample(n=40, random_state=42)).reset_index(drop=True)

# 分割訓練與測試集
X = df_sampled.drop(columns=['label'])
y = df_sampled['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 隨機森林
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

#ADA
adaboost = AdaBoostClassifier(estimator=DecisionTreeClassifier(), n_estimators=50, random_state=42)
adaboost.fit(X_train, y_train)
y_pred_adaboost = adaboost.predict(X_test)

# K-近鄰
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# LDA
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
y_pred_lda = lda.predict(X_test)

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

target_names = ['class 0', 'class 1']
print('random_forest:\n')
print(classification_report(y_test, y_pred_rf, target_names=target_names))
print('adaboost:\n')
print(classification_report(y_test, y_pred_adaboost, target_names=target_names))
print('KNN:\n')
print(classification_report(y_test, y_pred_knn, target_names=target_names))
print('LDA:\n')
print(classification_report(y_test, y_pred_lda, target_names=target_names))
print('Decision Tree:\n')
print(classification_report(y_test, y_pred_dt, target_names=target_names))