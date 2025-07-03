import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib

df = pd.read_csv('C:/Users/admin/Desktop/creditcard.csv/creditcard.csv')

X = df.drop('Class',axis =1).select_dtypes(include=[np.number])
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

smote = SMOTE(random_state=42)
X_train_resampled , y_train_resampled = smote.fit_resample(X_train,y_train)

scaler = StandardScaler()
X_train_resampled_scale = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)


joblib.dump(scaler,'C:/Users/admin/Desktop/fraud_scaler.pkl')
print("Scaler Saved Sucessfully✅")

