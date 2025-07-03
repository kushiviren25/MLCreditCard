import numpy as np 
import pandas as pd 
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score, precision_score,recall_score,accuracy_score

# Load dataset:
df = pd.read_csv('C:/Users/admin/Desktop/creditcard.csv/creditcard.csv')
print(df.head())
print(df['Class'].value_counts())

# Basic Preprocessing:
X = df.drop('Class', axis=1)
X = X.select_dtypes(include=[np.number]) 
y = df['Class']
print(X.dtypes)

# Train and Testing data:
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y, test_size=0.2)

# SMOTE: Handle Class Imbalance:
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# Feature Scaling the XTrainSMOTE and XTest:
scaler = StandardScaler()
X_train_res_scale = scaler.fit_transform(X_train_res) 
X_test_scale = scaler.transform(X_test)

# Visualization:
## Pie Chart:
fraud_counts = df['Class'].value_counts()
label = ['Fraud', 'Not Fraud']
color = ['Red', 'Green']
plt.pie(fraud_counts, labels=label, colors=color, autopct="%1.1f%%", startangle=140)
plt.title('Fraud vs Non Fraud')
plt.axis('equal')
plt.show()
plt.close()

## Histogram:
plt.figure(figsize=(8, 5))
plt.hist(df['Amount'], bins=20, color='blue', edgecolor='black')
plt.title('Transaction Amount')
plt.xlabel('Amount')
plt.ylabel('Frequency')
plt.show()
plt.close()

## Histogram 2D:
plt.hist2d(df['Amount'], df['Time'], bins=(100, 24), cmap='plasma')
plt.colorbar(label='Number of Transactions')
plt.title('Amount Vs Hours')
plt.xlabel('Amount')
plt.ylabel('Hours')
plt.show()
plt.close()

# Statistics:
print('Transaction Mean:', df['Amount'].mean())
print('Max Transaction:', df['Amount'].max())
print('Min Transaction:', df['Amount'].min())

# Models:
# 1. Logistic Regression:
log_model = LogisticRegression(max_iter=1000, class_weight='balanced')
log_model.fit(X_train_res_scale, y_train_res)
log_pred = log_model.predict(X_test_scale)

print("LOGISTIC REGRESSION:")
print("Classification Report:", classification_report(y_test, log_pred, zero_division=0))
print("Accuracy:", accuracy_score(y_test, log_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, log_pred))
print("F1 Score:", f1_score(y_test, log_pred))

# 2. Random Forest:
print("Fitting Random Forest model...")
rcf_model = RandomForestClassifier(n_estimators=20, max_depth=3, random_state=42)
rcf_model.fit(X_train_res_scale, y_train_res)
rcf_pred = rcf_model.predict(X_test_scale)

print('RANDOM FOREST CLASSIFIER:')
print("Classification Report:", classification_report(y_test, rcf_pred, zero_division=0))
print("Accuracy:", accuracy_score(y_test, rcf_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, rcf_pred))
print("F1 Score:", f1_score(y_test, rcf_pred))

# 3. XGBoost:
print("Fitting XGBoost model...")
xgb_model = XGBClassifier(random_state=42, min_child_weight=1,eval_metric='logloss')
xgb_model.fit(X_train_res_scale, y_train_res)
xgb_pred = xgb_model.predict(X_test_scale)

print('XGBOOST:')
print("Classification Report:", classification_report(y_test, xgb_pred, zero_division=0))
print("Accuracy:", accuracy_score(y_test, xgb_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, xgb_pred))
print("F1 Score:", f1_score(y_test, xgb_pred))

print("Total Number of Transactions :",sum(y_test==1))
print("Total Number of Fraud Transactions:",sum(y_test==1))
print("Total Number of Non Fraud Transactions",sum(y_test==0))

# Comparison :
models = ['Logistic Regression','Random Forest ','XGBoost']

f1 = [f1_score(y_test,log_pred),f1_score(y_test,rcf_pred),f1_score(y_test,xgb_pred)]
recall =[recall_score(y_test,log_pred),recall_score(y_test,rcf_pred),recall_score(y_test,xgb_pred)]
precision = [precision_score(y_test,log_pred),precision_score(y_test,rcf_pred),precision_score(y_test,xgb_pred)]
accuracy = [accuracy_score(y_test,log_pred),accuracy_score(y_test,rcf_pred),accuracy_score(y_test,xgb_pred)]


Result = pd.DataFrame(
    {
        'Models ': models,
        'F1_score' : f1,
        'Recall_Score' : recall,
        'Precision_Score':precision,
        'Accuracy_Score':accuracy
    }
)

print(Result.sort_values(by='F1_score',ascending = False))

print("Total No of Transactions :",len(y_test))
print("Actual Number of Fraud Cases :",sum(y_test==1))
print("Actual Number of Non Fraud Cases :",sum(y_test==0))

print("Predicted Number of Fraud Cases  by Log Regression:",sum(log_pred==1))
print("Predicted Number of Fraud Cases by Random Forest:",sum(rcf_pred==1))
print("Predicted Number of Fraud Cases by XGBoost:",sum(xgb_pred==1))

# Fraud Score and how safe are the transactions:

fraud_prob = xgb_model.predict_proba(X_test_scale)[:,1]

f_low = np.percentile(fraud_prob,33)
f_mid = np.percentile(fraud_prob,66)

fraud_score = []
actions = []

for prob in fraud_prob :
    if prob <= f_low:
        fraud_score.append("Low Fraud Score")
        actions.append("Transaction Approved")
    elif prob <= f_mid:
        fraud_score.append("Moderate Fraud Score")
        actions.append("Requires OTP verification")
    else:
        fraud_score.append("High Fraud Score")
        actions.append("Transaction Blocked ")

fraud_probability = pd.DataFrame({
    'Actual' : y_test,
    'Predicted' : xgb_pred,
    'Fraud Probability' : fraud_prob,
    'Fraud Score': fraud_score,
    'Recommended Actions ': actions
})

print(fraud_probability.sort_values(by='Fraud Probability',ascending= False))

print("Preview of all types of Risks :")
print(fraud_probability.groupby('Fraud Score').apply(lambda x : x.sort_values('Fraud Probability',ascending = False).head(3)))


print("Fraud Report :",fraud_probability['Fraud Score'].value_counts())



