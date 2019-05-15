# --------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Code starts here
df = pd.read_csv(path)
print(df.head())
print(df.info())

for col in ['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM','CLM_AMT']:
    df[col]  = df[col].str.replace('$','')
    df[col]  = df[col].str.replace(',','')

print(df.head())
X = df.drop('CLAIM_FLAG', axis=1)
y = df['CLAIM_FLAG'].copy()

count = df['CLAIM_FLAG'].value_counts()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 6)




# Code ends here


# --------------
# Code starts here
X_train[['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM','CLM_AMT']] = \
                    X_train[['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM','CLM_AMT']].astype(float)
X_test[['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM','CLM_AMT']] = \
                    X_test[['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM','CLM_AMT']].astype(float)

print(X_train.isnull().sum())
print(X_train.isnull().sum())
# Code ends here


# --------------
# Code starts here

X_train.dropna(axis=0, subset=['YOJ','OCCUPATION'], inplace=True)
X_test.dropna(axis=0, subset=['YOJ','OCCUPATION'], inplace=True)

y_train = y_train[X_train.index]
y_test = y_test[X_test.index]
print(y_train.shape)
for col in ['AGE','CAR_AGE','INCOME','HOME_VAL']:
    X_train[col].fillna(X_train[col].mean(), inplace=True)
    X_test[col].fillna(X_test[col].mean(), inplace=True)

# Code ends here


# --------------
from sklearn.preprocessing import LabelEncoder
columns = ["PARENT1","MSTATUS","GENDER","EDUCATION","OCCUPATION","CAR_USE","CAR_TYPE","RED_CAR","REVOKED"]

# Code starts here
for col in columns:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col].astype(str))
    X_test[col] = le.fit_transform(X_test[col].astype(str))

# Code ends here



# --------------
from sklearn.metrics import precision_score 
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression



# code starts here 
model  = LogisticRegression(random_state=6)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

score = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred) 
print("score: {}, precision: {}".format(score, precision))
# Code ends here


# --------------
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# code starts here
smote = SMOTE(random_state=9)
X_train, y_train = smote.fit_sample(X_train, y_train)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# Code ends here


# --------------
# Code Starts here
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

score = accuracy_score(y_pred, y_test)
print("score: {}".format(score))
# Code ends here


