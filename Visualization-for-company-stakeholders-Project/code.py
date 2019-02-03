# --------------
#Importing header files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv(path)

loan_status = data['Loan_Status'].value_counts()
print(loan_status)
plt.xlabel('Loan Status')
plt.ylabel('Count for Loan Status')
plt.title('Bar plot for Loan Status')
loan_status.plot.bar(rot=0)

#Code starts here


# --------------
  #Task 1 - Loan Status
 #Importing header files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

property_and_loan =  data.groupby(['Property_Area','Loan_Status'])
property_and_loan = property_and_loan.size().unstack()

property_and_loan.plot(kind='bar', stacked=False, rot=45)
plt.show()

#Code starts here


# --------------
#Code starts here

education_and_loan = data.groupby(['Education','Loan_Status'])
education_and_loan = education_and_loan.size().unstack()

education_and_loan.plot(kind='bar',stacked=True, rot=45)
plt.xlabel("Education Status")
plt.ylabel("Loan Status")


# --------------
#Code starts here

graduate = data[data['Education']=='Graduate']
not_graduate = data[data['Education']=='Not Graduate']
#print(graduate)
#print(graduate['LoanAmount'])
#print(type(graduate['LoanAmount']))
graduate['LoanAmount'].plot(kind='density', label='Graduate')
not_graduate['LoanAmount'].plot(kind='density', label='Not Graduate')

#Code ends here

#For automatic legend display
plt.legend()


# --------------
#Code starts here

fig,(ax_1,ax_2,ax_3) = plt.subplots(nrows=3, ncols=1)
plt.subplots_adjust(hspace=1.35)

ax_1.scatter(data['LoanAmount'], data['ApplicantIncome'])
ax_1.set_title("Applicant Income")

ax_2.scatter(data['LoanAmount'] ,data['CoapplicantIncome'])
ax_2.set_title('Coapplicant Income')

data['TotalIncome'] = data['ApplicantIncome'] + data['CoapplicantIncome']

ax_3.scatter(data['LoanAmount'], data['TotalIncome'])
ax_3.set_title('Total Income')


