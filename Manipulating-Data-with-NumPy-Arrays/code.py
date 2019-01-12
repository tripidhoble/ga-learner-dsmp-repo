# --------------
# Importing header files
import numpy as np

# Path of the file has been stored in variable called 'path'

#New record
new_record=[[50,  9,  4,  1,  0,  0, 40,  0]]

#Code starts here

data = np.genfromtxt(path, delimiter=",", skip_header=1)
print("data: \n",data )

census = np.concatenate((data,new_record), axis=0)
print("census:\n",census)



# --------------
#Code starts here
#print(census)
age = census[:,0]
print(age)

max_age = age.max()
print("max_age: ", max_age)
min_age = age.min()
print("min_age: ", min_age)
age_mean = age.mean()
print("age_mean: ",age_mean)
age_std = age.std() 
print("age_std: ",age_std)



# --------------
#Code starts here
race_0 = census[census[:,2]==0].astype(int)
print("race_0:\n",race_0)

race_1 = census[census[:,2]==1].astype(int)
print("race_1:\n",race_1)

race_2 = census[census[:,2]==2].astype(int)
print("race_2:\n",race_2)

race_3 = census[census[:,2]==3].astype(int)
print("race_3:\n",race_3)

race_4 = census[census[:,2]==4].astype(int)
print("race_4:\n",race_4)

len_0 = len(race_0)
print("len_0: ",len_0)

len_1 = len(race_1)
print("len_1: ",len_1)

len_2 = len(race_2)
print("len_2: ",len_2)

len_3 = len(race_3)
print("len_3: ",len_3)

len_4 = len(race_4)
print("len_4: ",len_4)

minority_race = 3
print("minority_race:", minority_race)







# --------------
#Code starts here
senior_citizens = census[census[:,0]>60].astype(int)
print("senior_citizens: \n", senior_citizens)

working_hours_sum = np.sum(senior_citizens[:,6], axis=0)
print("working_hours_sum: \n", working_hours_sum)

senior_citizens_len = len(senior_citizens) 
print("senior_citizens_len: ", senior_citizens_len)

avg_working_hours = working_hours_sum/senior_citizens_len
print("avg_working_hours: ", avg_working_hours)



# --------------
#Code starts here
high = census[census[:,1]>10]
print("high:\n",high) 

low = census[census[:,1]<=10]
print("low:\n",low) 

avg_pay_high = np.mean(high[:,7])
print("avg_pay_high: ", avg_pay_high)

avg_pay_low = np.mean(low[:,7])
print("avg_pay_low: ",avg_pay_low)


