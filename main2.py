
import pandas as pd
import numpy as np
from matplotlib import pyplot
from scipy import stats
import statsmodels.api
import statsmodels.stats.multitest
import xlwt
from xlwt import Workbook
wb = Workbook()
sheet1 = wb.add_sheet('new_Sheet')



data21 = pd.ExcelFile(r'./TRYERDATA225.xlsx')

df3 = pd.read_excel(data21,sheet_name='new_Sheet')

checker=[]
checker2=[]
checker3 = []
checker4=[]
for i in range(0,df3.shape[0]):
    row=df3.iloc[ i, 57:59]
    row2=df3.iloc[ i, 61:63]
    numpyRow = row.to_numpy()
    numpyRow2 = row2.to_numpy()

    print (numpyRow)
    print(numpyRow2)
    jjj
    if(numpyRow[0]!=numpyRow[1]): # we check if the true or false values of independant samples changes after correction and add it in an empty list called checker
        checker.append(df3.index[i])
    else: # we check if the true values of independant samples still true after correction and add it in an empty list called checker3
        checker3.append(df3.index[i])

    if(numpyRow2[0]!=numpyRow2[1]): # we check if the true or false values of paired samples changes after correction and add it in an empty list called checker2
        checker2.append(df3.index[i])
    else: # we check if the true values of paired samples still true after correction and add it in an empty list called checker4
        checker4.append(df3.index[i])

Unique_ind=[]
for i in checker: # we use this loop to get the names of distinct genes of indpenadant samples
    Unique_ind.append(df3.iloc[i,1])

print("No. of distinct genes of indpenadant samples:  %s"%len(Unique_ind))


Unique_Paired=[] # we use this loop to get the names of distinct genes of paired samples
for i in checker2:
    Unique_Paired.append(df3.iloc[i,1])

print("No. of distinct genes of paired samples:  %s"%len(Unique_Paired))


Common_ind=[]
for i in checker3:
    Common_ind.append(df3.iloc[i,1])

print("No. of common genes of independant samples:  %s"%len(Common_ind))

Common_Paired=[]
for i in checker4:
    Common_Paired.append(df3.iloc[i,1])

print("No. of common genes of paired samples:  %s"%len(Common_Paired))


for i in range(len(Unique_ind)):
    sheet1.write(i+1,0,Unique_ind[i])

for i in range(len(Unique_Paired)):
    sheet1.write(i+1,1,Unique_Paired[i])


wb.save('Unique.xlsx')