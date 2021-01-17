
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



data21 = pd.ExcelFile(r'E:\venv\Biostatistics\Tryer number 006\Hypothesis2\TRYERDATA225.xlsx')

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

    if(numpyRow[0]!=numpyRow[1]):
        checker.append(df3.index[i])
    else:
        checker3.append(df3.index[i])

    if(numpyRow2[0]!=numpyRow2[1]):
        checker2.append(df3.index[i])
    else:
        checker4.append(df3.index[i])

Unique_ind=[]
for i in checker:
    Unique_ind.append(df3.iloc[i,1])

print(len(Unique_ind))


Unique_Paired=[]
for i in checker2:
    Unique_Paired.append(df3.iloc[i,1])

print(len(Unique_Paired))


Common_ind=[]
for i in checker3:
    Common_ind.append(df3.iloc[i,1])

print(len(Common_ind))

Common_Paired=[]
for i in checker4:
    Common_Paired.append(df3.iloc[i,1])

print(len(Common_Paired))
for i in range(len(Unique_ind)):
    sheet1.write(i+1,0,Unique_ind[i])

for i in range(len(Unique_Paired)):
    sheet1.write(i+1,1,Unique_Paired[i])


wb.save('Unique.xlsx')
