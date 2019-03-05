# --------------
#Importing header files
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#Code starts here

data = pd.read_csv(path)
#sns.distplot(data['Rating'], kde=False)
#plt.hist(data['Rating'], bins=10)
data.Rating.hist(bins=10)
plt.show()

data = data[data['Rating']<=5]
data.Rating.hist(bins=10)
plt.show()
#Code ends here


# --------------
# code starts here
import pandas as pd
total_null = data.isnull().sum()
#print("total_null: \n",total_null)
percent_null = (total_null/(data.isnull().count())) * 100
#print("percent_null: \n",percent_null)
missing_data = pd.concat([total_null,percent_null], keys=['Total','Percent'],axis=1)
print(missing_data)

data.dropna(inplace=True)
total_null_1 = data.isnull().sum()
percent_null_1 = (total_null_1/(data.isnull().count())) * 100
missing_data_1 = pd.concat([total_null_1,percent_null_1], keys=['Total','Percent'],axis=1)
print(missing_data_1)
# code ends here


# --------------

#Code starts here
sns.catplot(x='Category', y="Rating", data=data, kind="box", height=10)
plt.xticks(rotation=90)
plt.title("Rating vs Category [BoxPlot]")
plt.show()
#Code ends here


# --------------
#Importing header files
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

#Code starts here
#print(data['Installs'].value_counts)
data['Installs'] = data['Installs'].str.replace(',','')
data['Installs'] = data['Installs'].str.replace('+','')
data['Installs'] = data['Installs'].astype(int)

le = LabelEncoder()
data['Installs'] = le.fit_transform(data['Installs'])
#plt.regplot(x='Intalls', y='Rating', data=data)
sns.jointplot('Rating', 'Installs', data=data, kind='reg')
plt.title("Rating vs installs [Regplot]")

#Code ends here



# --------------
#Code starts here
print(data['Price'].value_counts())
data['Price'] = data['Price'].str.replace('$','')
#print(data.tail())
data['Price'] = data['Price'].astype(float)
sns.regplot(x='Price', y='Rating', data=data)
plt.title(" Rating vs Price [Regplot]")
#Code ends here


# --------------

#Code starts here

print(data['Genres'].unique())
#print(data.head())
data['Genres'] = data['Genres'].str.split(';').str[0]
#print(data.head())
gr_mean = data[['Genres','Rating']].groupby(['Genres'], as_index=False).mean()
gr_mean = gr_mean.sort_values(by='Rating')
print("First: \n",gr_mean.head(1))
print("Last: \n",gr_mean.tail(1))



#Code ends here


# --------------

#Code starts here
print(data['Last Updated'])

data['Last Updated'] = pd.to_datetime(data['Last Updated'])
max_date = data['Last Updated'].max()

data['Last Updated Days'] = (max_date - data['Last Updated']).dt.days 

print(data[['Last Updated','Last Updated Days']])

sns.regplot(x='Last Updated Days', y='Rating', data=data)
plt.title("Rating vs Last Updated [Regplot] ")

#Code ends here


