# --------------
import pandas as pd
import scipy.stats as stats
import math
import numpy as np
import warnings

warnings.filterwarnings('ignore')
#Sample_Size
sample_size=2000

#Z_Critical Score
z_critical = stats.norm.ppf(q = 0.95)  


# path        [File location variable]
data = pd.read_csv(path)
#print(data.head())

#Code starts here

data_sample = data.sample(n=sample_size, random_state=0)
#print(sample_data.shape)

sample_mean = data_sample['installment'].mean()
print("sample_mean: ",sample_mean)
sample_std = data_sample['installment'].std()

margin_of_error = z_critical * (sample_std / math.sqrt(sample_size))
print(margin_of_error)

confidence_interval = ((sample_mean-margin_of_error),(sample_mean+margin_of_error))
print("confidence_interval: ",confidence_interval)

true_mean = data['installment'].mean()
print("true_mean: ",true_mean)

#print(confidence_of_interval[0])
if confidence_interval[0] <= true_mean <=confidence_interval[1]:
    print('Yes')





# --------------
import matplotlib.pyplot as plt
import numpy as np

#Different sample sizes to take
sample_size=np.array([20,50,100])

#Code starts here
fig ,axes = plt.subplots(nrows=3, ncols=1)

for i in range(len(sample_size)):
    m=[]
    for j in range(1000):
        sample_data = data.sample(n=sample_size[i], random_state=0)
        m.append(sample_data['installment'].mean())
    mean_series = pd.Series(m)
    #print(mean_series)
    axes[i].plot(mean_series)



# --------------
#Importing header files

from statsmodels.stats.weightstats import ztest

#Code starts here
data['int.rate'] = data['int.rate'].astype(str).str[:-1].astype(np.float)
data['int.rate'] = data['int.rate']/100
print("mean: ",data['int.rate'].mean())
z_statistic, p_value =ztest(data[data['purpose']=='small_business']['int.rate'], value=data['int.rate'].mean(), alternative='larger')
print("z_statistic: ",z_statistic)
print("p_value: ",p_value)
if(p_value<0.05):
    print("Reject Null hypothesis")
else:
    print("Accept Null hypothesis")


# --------------
#Importing header files
from statsmodels.stats.weightstats import ztest

#Code starts here
z_statistic, p_value = ztest(x1 = data[data['paid.back.loan']=='No']['installment'], x2 = data[data['paid.back.loan']=='Yes']['installment'])
print("z_statistic: ",z_statistic)
print("p_value: ",p_value)

if(p_value < 0.05):
    print("Rejected Null hypothesis" )
else:
    print("Accepted Null hypothesis" )



# --------------
#Importing header files
from scipy.stats import chi2_contingency

#Critical value 
critical_value = stats.chi2.ppf(q = 0.95, # Find the critical value for 95% confidence*
                      df = 6)   # Df = number of variable categories(in purpose) - 1

#Code starts here

yes = data[data['paid.back.loan']=='Yes']['purpose'].value_counts()
print('yes: \n',yes)

no = data[data['paid.back.loan']=='No']['purpose'].value_counts()
print('no: \n',no)

observed = pd.concat([yes.transpose(),no.transpose()], axis=1, keys=['Yes','No'])
print("observed: \n",observed)

chi2, p, dof, ex = chi2_contingency(observed)

print("chi2: ",chi2)
print("critical_value: ",critical_value)

if(chi2>critical_value):
    print("Reject Null hypothesis")
else:
    print("Accept Null hypothesis")



