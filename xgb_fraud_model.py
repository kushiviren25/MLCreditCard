import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, roc_auc_score , confusion_matrix
import joblib

# 1. Load data
df = pd.read_csv('C:/Users/admin/Desktop/creditcard.csv/creditcard.csv')


# 2. Feature & target separation
X = df.drop('Class', axis=1)
X = X.select_dtypes(include=[np.number])
y = df['Class']

# Feature Order :
feature_order = X.columns.tolist()
joblib.dump(feature_order,'C:/Users/admin/Desktop/feature_order.pkl')

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Scaling :
scaler = StandardScaler()
X_train_scale  = scaler.fit_transform(X_train_resampled)
X_test_scale = scaler.transform(X_test)



# Call the model :
xgb_model = XGBClassifier(random_state =42 ,eval_metric ='logloss')

#4. Fit the model :
xgb_model.fit(X_train_scale,y_train_resampled)

#5. Predict the model :
xgb_pred = xgb_model.predict(X_test_scale)
y_proba = xgb_model.predict_proba(X_test_scale)[:, 1]


#Evaluation Metrics :
print("Classification Report :",classification_report(y_test,xgb_pred))
print("Confusion Matrix :",confusion_matrix(y_test,xgb_pred))
print("ROC AUC Score :",roc_auc_score(y_test,xgb_pred))


# Save the load 
joblib.dump(xgb_model,'C:/Users/admin/Desktop/xgb_fraud_model.pkl')
joblib.dump(scaler,'C:/Users/admin/Desktop/fraud_scaler.pkl')
joblib.dump(list(X.columns),'C:/Users/admin/Desktop/feature_order.pkl')

# Test on some known fraud cases
fraud_cases = df[df['Class'] == 1].sample(n=5, random_state=42)
fraud_features = fraud_cases.drop('Class', axis=1)

# Scale using the same scaler
fraud_scaled = scaler.transform(fraud_features)

# Predict
probs = xgb_model.predict_proba(fraud_scaled)
preds = xgb_model.predict(fraud_scaled)

for i, (p, pred) in enumerate(zip(probs, preds)):
    print(f"Test {i+1}: Prob = {p[1]:.4f}, Prediction = {int(pred)}")



print("Model and Scaler saved successfully! ✅")