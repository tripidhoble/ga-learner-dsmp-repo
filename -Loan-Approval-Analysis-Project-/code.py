# --------------
# Import packages
import numpy as np
import pandas as pd
from scipy.stats import mode 
 
bank = pd.read_csv(path)
#print(bank.head())
#print(bank.head())
categorical_var = bank.select_dtypes(include = 'object')
print("categorical_var: \n",categorical_var.head())

numerical_var = bank.select_dtypes(include = 'number')
print("numerical_var: \n", numerical_var.head())

# code starts here






# code ends here


# --------------
# code starts here
print(bank.head())

banks = bank.drop(['Loan_ID'], axis=1)
print(banks.head())

print(banks.isnull().sum())

bank_mode = banks.mode()
print(bank_mode)

#print(banks.columns)
#banks.columns.fillna(bank_mode, inplace=True )
#print(banks.head())
for column in banks.columns:
    banks[column].fillna(banks[column].mode()[0], inplace=True)

#code ends here


# --------------
# Code starts here

avg_loan_amount = banks.pivot_table(index=['Gender','Married','Self_Employed'], values='LoanAmount' ,aggfunc=np.mean)

print(avg_loan_amount)

# code ends here



# --------------
# code starts here
print(banks.shape)
loan_approved_se = banks[(banks['Self_Employed']=='Yes') & (banks['Loan_Status']=='Y')].count()[0]
print("loan_approved_se: ", loan_approved_se)

loan_approved_nse = banks[(banks['Self_Employed']=='No') & (banks['Loan_Status']=='Y')].count()[0]
print("loan_approved_nse: ", loan_approved_nse)

Loan_Status = 614

percentage_se = (loan_approved_se/Loan_Status)*100
print("percentage_se: ",percentage_se)
percentage_nse = (loan_approved_nse/Loan_Status)*100
print("percentage_nse: ",percentage_nse)

# code ends here


# --------------
# code starts here

loan_term= banks['Loan_Amount_Term'].apply(lambda months : months/12 )
#print(loan_term)
#print(type(loan_term))

big_loan_term = loan_term[loan_term >= 25].size
print(big_loan_term)

# code ends here



# --------------
# code starts here

loan_groupby = banks.groupby('Loan_Status')
print(loan_groupby.groups)

loan_groupby[['ApplicantIncome','Credit_History']]
print(loan_groupby.groups)

mean_values = loan_groupby.mean()
print("mean_values: ",mean_values)
print(type(mean_values))
# code ends here


