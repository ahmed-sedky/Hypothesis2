# # from numpy import mean
# # from numpy import std
# # from numpy.random import randn
# # from numpy.random import seed

# import pandas as pd  # pandas to read excel file
# import numpy as np  # use it to make sum or mean
# data = pd.ExcelFile(r'C:\Users\Hp\Downloads\hypothesis\lusc-rsem-fpkm-tcga-t_paired.xlsx')  # open file Excel
# print(data.sheet_names)
# # df = pd.read_excel(data,sheet_name='lusc-rsem-fpkm-tcga-t_paired')  # to get the data from exel
# df = pd.read_excel(data, sheet_name='lusc-rsem-fpkm-tcga-t_paired') # to get the data from exel
# # print(df.iloc[0])
# # print(df.sum(axis = 0, skipna = True))

# # print(rowData.to_numpy())
# Means=[]
# for i in range(0,19647):
#     rowData = df.iloc[ i, 1: ]  #method to get location in read_excel in pandas
#     Means.append(np.sum(rowData.to_numpy())/float(51))
# print(Means)

# df['Means']=Means
# Writer=pd.ExcelWriter('NewTryer2.xlsx') # pylint: disable=abstract-class-instantiated
# df.to_excel(Writer,'new_Sheet')
# Writer.save()



# # a=np.array(df[0])
# # b=np.array(df[1])
# # print(np.concatenate((a,b),axis = 0))



# # # seed random number generator
# # seed(1)
# # # prepare data
# # data1 = 20 * randn(1000) + 100
# # data2 = data1 + (10 * randn(1000) + 50)
# # # summarize
# # print('data1: mean=%.3f stdv=%.3f' % (mean(data1), std(data1)))
# # print('data2: mean=%.3f stdv=%.3f' % (mean(data2), std(data2)))
# # # plot
# # pyplot.scatter(data1, data2)
# # pyplot.show()

import pandas as pd
import numpy as np
from matplotlib import pyplot
from scipy import stats
import statsmodels.api 
import statsmodels.stats.multitest 



data = pd.ExcelFile(r'E:\venv\Biostatistics\Tryer number 006\Hypothesis2\Tryer44.xlsx')

df = pd.read_excel(data,sheet_name='lusc-rsem-fpkm-tcga_paired')

data2 = pd.ExcelFile(r'E:\venv\Biostatistics\Tryer number 006\Hypothesis2\TRYERDATA.xlsx')
print(data2.sheet_names)
df2 = pd.read_excel(data2,sheet_name='DATA NUMBERS')
# df = df.drop(df.index[0])
# print(df.iloc[0])
# print(df.sum(axis = 0, skipna = True))
print(df.shape[0])
# print(rowData.to_numpy())
rows = df.shape[0]
Correlations = []
i = 0
while (i < rows):
    if i >= df.shape[0]:
        break
    rowData = df.iloc[ i, 2:]#i=0
    rowData2 = df2.iloc[ i, 2:]
    numpyRow = rowData.to_numpy()
    numpyRow2 = rowData2.to_numpy()
    numpyRow = np.where(numpyRow <= 5, 0, numpyRow)
    numpyRow2 = np.where(numpyRow2 <= 5, 0, numpyRow2)
    if (np.count_nonzero(numpyRow) < len(numpyRow) * 0.5) or (np.count_nonzero(numpyRow2) < len(numpyRow2) * 0.5):
        df = df.drop([df.index[i]])
        df2 = df2.drop(df2.index[i])
        continue
    X=np.corrcoef((rowData.to_numpy()).astype(float),(rowData2.to_numpy()).astype(float))[0, 1]
    # np.cov(X.astype(float))  # works
    Correlations.append(float(X))
    i+=1

# print(Correlations)
# z =df.shape[0]
# Correlations=[]
# for i in range(0,z):
#     rowData = df.iloc[ i, 2: ]#i=0
#     rowData2 = df2.iloc[ i, 2: ]
#     for x in rowData:
#         count1 = 0 
#         if (x == 0):
#             count1 = count1 + 1
#     for a in rowData2:
#         count2 = 0
#         if (a == 0):
#             count2 = count2 + 1
#     if count1 > 25:
#         name =df.iloc [i,0]
#         print(df.index[i])
#         df = df.drop([name] )
#         df2 = df2.drop([name])
#         # i = i+1
#         # z = z -1
#     else:
#         X=np.corrcoef((rowData.to_numpy()).astype(float),(rowData2.to_numpy()).astype(float)) [0,1]
#         # np.cov(X.astype(float))  # works
#         Correlations.append(float(X))
# # print(Correlations)



print('=' * 100)
print (max(Correlations))
index_max = Correlations.index(max(Correlations))

print(df.iloc[index_max ,0])
print('=' * 100)
print (min(Correlations))
index_min = Correlations.index(min(Correlations))
print(df.iloc[index_min ,0])
print(index_max)
print(index_min)



df['Correlations']=Correlations
Writer=pd.ExcelWriter(r'E:\venv\Biostatistics\Tryer number 006\Hypothesis2\TRYERDATA225.xlsx')
df.to_excel(Writer,'new_Sheet')
Writer.save()

df['Sorted_Correlations']=np.sort(Correlations)
Writer=pd.ExcelWriter(r'./NewTryer309.xlsx')
df.to_excel(Writer,'new_Sheet')
Writer.save()
for i in range(0, df.shape[0]):
    if((i==index_max) or (i==index_min)):
        rowData = df.iloc[i, 2:df2.shape[1]]  # i=0
        rowData2 = df2.iloc[i, 2:df2.shape[1]]
        pyplot.scatter((rowData.to_numpy()).astype(float),(rowData2.to_numpy()).astype(float))
        pyplot.show()

ind = []
ind_pval = []
for i in range(0,df.shape[0]):
    rowData = df.iloc[ i, 2: ]#i=0
    rowData2 = df2.iloc[ i, 2: ]
    
    ind.append(stats.ttest_ind((rowData.to_numpy()).astype(float),(rowData2.to_numpy()).astype(float)))
    ind_pval.append(ind[i][1])
    # print(t_test_ind)
# print(t_test_ind_pval)
df['ind_pval'] = ind_pval
Writer=pd.ExcelWriter(r'E:\venv\Biostatistics\Tryer number 006\Hypothesis2\TRYERDATA225.xlsx')
df.to_excel(Writer,'new_Sheet')
Writer.save()


print ('=' * 150)

ind_corr = []
ind_corr_pValue = []
ind_pval_reject = []
for i in range(0,df.shape[0]):
    rowData = df.iloc[ i, 2: ]#i=0
    rowData2 = df2.iloc[ i, 2: ]
    ind_corr.append(statsmodels.stats.multitest.multipletests(ind_pval[i], alpha=0.05, method='fdr_tsbky', is_sorted=False, returnsorted=False))
    # print(t_test_ind_corr_p_val[i][1][0])
    ind_corr_pValue.append(ind_corr[i][1][0])
    ind_pval_reject.append(ind_corr[i][0][0])

df['ind_corr_pval'] = ind_corr_pValue
Writer=pd.ExcelWriter(r'E:\venv\Biostatistics\Tryer number 006\Hypothesis2\TRYERDATA225.xlsx')
df.to_excel(Writer,'new_Sheet')
Writer.save()

df['reject_null_hypothesis_ind'] = ind_pval_reject
Writer=pd.ExcelWriter(r'./NewTryer309.xlsx')
df.to_excel(Writer,'new_Sheet')
Writer.save()

ind_corr_p_val2 = []
ind_corr_pval_reject = []
for i in range(0,df.shape[0]):
    rowData = df.iloc[ i, 2: ]#i=0
    rowData2 = df2.iloc[ i, 2: ]
    ind_corr_p_val2.append(statsmodels.stats.multitest.multipletests(ind_corr_pValue[i], alpha=0.05, method='fdr_tsbky', is_sorted=False, returnsorted=False))
    # print(t_test_ind_corr_p_val[i][1][0])
    ind_corr_pval_reject.append(ind_corr_p_val2[i][0][0])

df['reject_null_hypothesis_ind _corrected'] = ind_corr_pval_reject
Writer=pd.ExcelWriter(r'E:\venv\Biostatistics\Tryer number 006\Hypothesis2\TRYERDATA225.xlsx')
df.to_excel(Writer,'new_Sheet')
Writer.save()

paired = []
paired_pval = []
for i in range(0,df.shape[0]):
    rowData = df.iloc[ i, 2: 50]#i=0
    rowData2 = df2.iloc[ i, 2:50 ]
    paired.append(stats.ttest_rel((rowData.to_numpy()).astype(float),(rowData2.to_numpy()).astype(float)))
    paired_pval.append(paired[i][1])


df['paired_pval'] = paired_pval
Writer=pd.ExcelWriter(r'E:\venv\Biostatistics\Tryer number 006\Hypothesis2\TRYERDATA225.xlsx')
df.to_excel(Writer,'new_Sheet')
Writer.save()

print ('=' * 150)
paired_corr = []
paired_corr_pValue = []
paired_reject = []
for i in range(0,df.shape[0]):
    rowData = df.iloc[ i, 2: ]#i=0
    rowData2 = df2.iloc[ i, 2: ]
    paired_corr.append(statsmodels.stats.multitest.multipletests(paired_pval[i], alpha=0.05, method='fdr_tsbky', is_sorted=False, returnsorted=False))
    # print(t_test_rel_corr_p_val[i][1][0])
    paired_corr_pValue.append(paired_corr[i][1][0])
    paired_reject.append(paired_corr[i][0][0])

df['paired_pval_corr'] = paired_corr_pValue
Writer=pd.ExcelWriter(r'E:\venv\Biostatistics\Tryer number 006\Hypothesis2\TRYERDATA225.xlsx')
df.to_excel(Writer,'new_Sheet')
Writer.save()

df['reject_null_hypothesis_paired'] = paired_reject
Writer=pd.ExcelWriter(r'./NewTryer309.xlsx')
df.to_excel(Writer,'new_Sheet')
Writer.save()

paired_corr2 = []
paired_corr_pval_reject = []
for i in range(0,df.shape[0]):
    rowData = df.iloc[ i, 2: ]#i=0
    rowData2 = df2.iloc[ i, 2: ]
    paired_corr2.append(statsmodels.stats.multitest.multipletests(paired_corr_pValue[i], alpha=0.05, method='fdr_tsbky', is_sorted=False, returnsorted=False))
    paired_corr_pval_reject.append(paired_corr2[i][0][0])

df['reject_null_hypothesis_paired_corrected'] = paired_corr_pval_reject
Writer=pd.ExcelWriter(r'E:\venv\Biostatistics\Tryer number 006\Hypothesis2\TRYERDATA225.xlsx')
df.to_excel(Writer,'new_Sheet')
Writer.save()


num_true_ind_before = np.sum(ind_pval_reject)
num_true_ind_after = np.sum(ind_corr_pval_reject)

num_true_paired_before = np.sum(paired_reject)
num_true_paired_after = np.sum(paired_corr_pval_reject)

print ( num_true_ind_before )
print ( num_true_ind_after )
print( num_true_paired_before )
print ( num_true_paired_after )





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






df['Unique_ind'] = Unique_ind
df['Unique_Paired'] = Unique_Paired
Writer=pd.ExcelWriter(r'E:\venv\Biostatistics\Tryer number 006\Hypothesis2\TRYERDATA225.xlsx')
df.to_excel(Writer,'new_Sheet2')
Writer.save()


