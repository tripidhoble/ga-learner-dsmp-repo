# --------------
import pandas as pd
from sklearn import preprocessing

#path : File path

# Code starts here

# read the dataset
dataset = pd.read_csv(path)

# look at the first five columns
print(dataset.head())

# Check if there's any column which is not useful and remove it like the column id
dataset.drop(['Id'],axis = 1, inplace = True)

# check the statistical description
print(dataset.describe())


# --------------
# We will visualize all the attributes using Violin Plot - a combination of box and density plots
import seaborn as sns
from matplotlib import pyplot as plt

#names of all the attributes 
cols = dataset.columns.values

#number of attributes (exclude target)
size = len(cols) - 1

#x-axis has target attribute to distinguish between classes
x = dataset['Cover_Type'].copy()

#y-axis shows values of an attribute
y = dataset.drop(['Cover_Type'], axis = 1)

#Plot violin for all attributes
#for i in range(size):



# --------------
import numpy
threshold = 0.5

# no. of features considered after ignoring categorical variables

num_features = 10

# create a subset of dataframe with only 'num_features'
subset_train = dataset.iloc[:,:num_features]
cols = subset_train.columns
print(cols)
#Calculate the pearson co-efficient for all possible combinations
data_corr = subset_train.corr()
sns.heatmap(data_corr, annot=True, cmap="YlGnBu")
plt.show()

# Set the threshold and search for pairs which are having correlation level above threshold
corr_var_list = []
for i in range(1,len(cols)):
    for j in range(i):
        if((abs(data_corr.iloc[i,j]) > threshold) & (abs(data_corr.iloc[i,j]) < 1)):
            corr_var_list.append([data_corr.iloc[i,j], i, j])
#print(corr_var_list)

s_corr_list = sorted(corr_var_list ,key = lambda x:x[0])

#print(s_corr_list)

for corr_value, i, j in s_corr_list:
    print("corr_value:%s,  column1: %s, column2: %s" %(corr_value, cols[i], cols[j]))




#corr_var_list = corr_var_list[abs(corr_var_list) > threshold].index.values

#print(corr_var_list)
#print(type(corr_var_list))

# Sort the list showing higher ones first 


#Print correlations and column names

#https://stackoverflow.com/questions/29294983/how-to-calculate-correlation-between-all-columns-and-remove-highly-correlated-on


# --------------
#Import libraries 
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler
import numpy as np

# Identify the unnecessary columns and remove it 
dataset.drop(columns=['Soil_Type7', 'Soil_Type15'], inplace=True)

X = dataset.drop(['Cover_Type'],axis=1)
Y = dataset['Cover_Type'].copy()

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Scales are not the same for all variables. Hence, rescaling and standardization may be necessary for some algorithm to be applied on it.
#Standardized
#Apply transform only for non-categorical data
std_model = StandardScaler()
X_train_temp = std_model.fit_transform(X_train.iloc[:,:10])
X_test_temp = std_model.fit_transform(X_test.iloc[:,:10])

#Concatenate non-categorical data and categorical
X_train1 = np.concatenate((X_train_temp, X_train.iloc[:,10:]), axis = 1)
X_test1 = np.concatenate((X_test_temp, X_test.iloc[:,10:]), axis = 1)

print(type(X_train1))

scaled_features_train_df = pd.DataFrame(X_train1, index = X_train.index, columns = X_train.columns)
scaled_features_test_df = pd.DataFrame(X_test1, index = X_test.index, columns = X_test.columns)

print(scaled_features_train_df.head())
print(scaled_features_test_df.head())









# --------------
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif


# Write your solution here:
skb = SelectPercentile(score_func = f_classif, percentile = 20)

predictors = skb.fit_transform(X_train1, Y_train)

scores = skb.scores_
print(scores)
print(X_train1)
top_k_index = skb.get_support(True)
top_k_predictors = X_train.columns[top_k_index]
print("top_k_index: ", top_k_index)
print("top_k_predictors: ",top_k_predictors)






# --------------
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score

clf = OneVsRestClassifier(LogisticRegression())
clf1 = OneVsRestClassifier(LogisticRegression()) 

model_fit_all_features = clf1.fit(X_train, Y_train)
predictions_all_features = model_fit_all_features.predict(X_test)
score_all_features = accuracy_score(Y_test, predictions_all_features)
print(score_all_features)

model_fit_top_features = clf.fit(scaled_features_train_df [top_k_predictors], Y_train)
predictions_top_features = model_fit_top_features.predict(scaled_features_test_df [top_k_predictors])
score_top_features = accuracy_score(Y_test, predictions_top_features)
print(score_top_features)




