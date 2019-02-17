# --------------
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
# code starts here
df = pd.read_csv(path)
print(df.head())

X = df[['ages','num_reviews','piece_count','play_star_rating','review_difficulty','star_rating','theme_name','val_star_rating','country']] 
y = df['list_price']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3, random_state=6)


# code ends here



# --------------
import matplotlib.pyplot as plt

# code starts here        
cols = X_train.columns.values

fig ,axes = plt.subplots(nrows = 3 , ncols = 3)

for i in range(3):
    for j in range(3):
        col = cols[i*3 + j]
        axes[i,j].scatter(X_train[col],y_train)
        axes[i,j].set_xlabel(col)
        axes[i,j].set_ylabel(y_train.name)
plt.tight_layout()
plt.show()       

# code ends here



# --------------
# Code starts here
corr = pd.DataFrame(index=cols,columns=cols)

for i in cols:
    for j in cols:
        corr.loc[i,j] = X_train[i].corr(X_train[j])
print(corr)

X_train.drop(columns=['play_star_rating','val_star_rating'],inplace=True)
X_test.drop(columns=['play_star_rating','val_star_rating'],inplace=True)
print("X_train:\n",X_train.head())
print("X_test:\n",X_test.head())
# Code ends here


# --------------
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Code starts here
regressor = LinearRegression()

regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)

mse = mean_squared_error(y_test,y_pred)
print('mse: ',mse)
r2 = r2_score(y_test,y_pred)
print("r2: ",r2)

# Code ends here


# --------------
# Code starts here

residual = y_test - y_pred

plt.hist(residual)
plt.show()


# Code ends here


