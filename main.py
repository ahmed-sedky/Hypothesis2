
import pandas as pd   # to deal with excel files read and write
import numpy as np     # library to deal with multi dimensional arrays with high high level mathematical functions to operate on these arrays
from matplotlib import pyplot   # we use it to plot max & min correlation
from scipy import stats     # we use it to calculate t_test which is a method init
import statsmodels.api 
import statsmodels.stats.multitest # we use it to make FDR correction on pvalue and to tell us about to regect or acceot null hypothesis
import xlwt  #write on new excel file
from xlwt import Workbook 
wb = Workbook()
sheet1 = wb.add_sheet('new_Sheet')


data = pd.ExcelFile(r'./Tryer44.xlsx')   # the file of expression levels on healthy tissue

df = pd.read_excel(data,sheet_name='lusc-rsem-fpkm-tcga_paired')

data2 = pd.ExcelFile(r'./TRYERDATA.xlsx')  # the file of expression levels on cancerous tissue
print(data2.sheet_names)
df2 = pd.read_excel(data2,sheet_name='DATA NUMBERS')

print("No. of Genes In Our Data:  %s" %df.shape[0])

rows = df.shape[0]
Correlations = [] 
i = 0
while (i < rows):
    if i >= df.shape[0]:
        break
    rowData = df.iloc[ i, 2:] # to read Data Row By Row
    rowData2 = df2.iloc[ i, 2:]
    numpyRow = rowData.to_numpy() # to store array as an numpy array
    numpyRow2 = rowData2.to_numpy()
    numpyRow = np.where(numpyRow <= 5, 0, numpyRow)   # we assume that the expression levels less than 5 is in accurate data so we replace it with zeros so we can count it and delete the row if they exceed 50% of The samples in the gene
    numpyRow2 = np.where(numpyRow2 <= 5, 0, numpyRow2)
    if (np.count_nonzero(numpyRow) < len(numpyRow) * 0.5) or (np.count_nonzero(numpyRow2) < len(numpyRow2) * 0.5): # delete the row if the zeros exceed 50% of The samples in the gene
        df = df.drop([df.index[i]])
        df2 = df2.drop(df2.index[i])
        continue
    X=np.corrcoef((numpyRow).astype(float),(numpyRow2).astype(float))[0, 1]
    Correlations.append(float(X)) # we append the correlation numbers on an empty list
    i+=1


print('=' * 100)
print ("max Correlation:  %s" %max(Correlations))
index_max = Correlations.index(max(Correlations))

print("name Of Gene That Has Max Correlation:  %s" %df.iloc[index_max ,0])
print('=' * 100)
print ("min Correlation:  %s" %min(Correlations))
index_min = Correlations.index(min(Correlations))
print("name Of Gene That Has Min Correlation:  %s" %df.iloc[index_min ,0])
print("index of max Correlation:  %s" %index_max)
print("index of min Correlation:  %s" %index_min)



df['Correlations']=Correlations # add correlation coulmn in excel that have correlation numbers of genes
Writer=pd.ExcelWriter(r'./TRYERDATA225.xlsx')
df.to_excel(Writer,'new_Sheet')
Writer.save()

df['Sorted_Correlations']=np.sort(Correlations) # add sorted correlation coulmn in excel
Writer=pd.ExcelWriter(r'./NewTryer309.xlsx')
df.to_excel(Writer,'new_Sheet')
Writer.save()

for i in range(0, df.shape[0]): # this for loop is to fetch the data of genes that have max & min Correlations ANd plot it
    if((i==index_max) or (i==index_min)):
        rowData = df.iloc[i, 2:df2.shape[1]]  # i=0
        rowData2 = df2.iloc[i, 2:df2.shape[1]]
        pyplot.scatter((rowData.to_numpy()).astype(float),(rowData2.to_numpy()).astype(float))
        pyplot.show()

ind = []
ind_pval = []
for i in range(0,df.shape[0]): # this for Loop Is To Get Pvalues of independant samples
    rowData = df.iloc[ i, 2: ] # we pass The First And Second Column
    rowData2 = df2.iloc[ i, 2: ]
    
    ind.append(stats.ttest_ind((rowData.to_numpy()).astype(float),(rowData2.to_numpy()).astype(float))) # the method ttest_ind returns two value 1st :statistics 2nd : pvalue so we Sort it in an empty array called  ind
    ind_pval.append(ind[i][1]) # we get only pvalue from array ind  and append it in an ind_pval array
    
df['ind_pval'] = ind_pval # we add column in excel called ind_pval that have independant pvalues
Writer=pd.ExcelWriter(r'./TRYERDATA225.xlsx')
df.to_excel(Writer,'new_Sheet')
Writer.save()


print ('=' * 150)

ind_corr = []
ind_corr_pValue = []
ind_pval_reject = []
for i in range(0,df.shape[0]):   # we get from this for loop the corrected pvalues of ind. samples and whether to regect or accept null hypothesis
    rowData = df.iloc[ i, 2: ]
    rowData2 = df2.iloc[ i, 2: ]
    # multipletests methos takes pvalue and alpha and the method we will use and returns 4 things: 1st whether to regect null hypothesis "True" or accept "False" and this is in list 2nd: the corrected pvalue in list 3rd: alphacSidak 4th:alphacBonf
    ind_corr.append(statsmodels.stats.multitest.multipletests(ind_pval[i], alpha=0.05, method='fdr_tsbky', is_sorted=False, returnsorted=False))

    ind_corr_pValue.append(ind_corr[i][1][0]) # we  get  the corrected pvalue and store it in an empty arrary called ind_corr_pValue
    ind_pval_reject.append(ind_corr[i][0][0]) # we  get  whether to regect null hypothesis "True" or accept "False" and store it in an empty arrary called ind_pval_regect

df['ind_corr_pval'] = ind_corr_pValue # we add new column with independant corrected pvalues
Writer=pd.ExcelWriter(r'./TRYERDATA225.xlsx')
df.to_excel(Writer,'new_Sheet')
Writer.save()

df['reject_null_hypothesis_ind'] = ind_pval_reject # we add new column with reject_null_hypothesis_ind
Writer=pd.ExcelWriter(r'./NewTryer309.xlsx')
df.to_excel(Writer,'new_Sheet')
Writer.save()

ind_corr_p_val2 = []
ind_corr_pval_reject = []
for i in range(0,df.shape[0]): # we use this for loop to get only whether to regect null hypothesis or accept it after correction of ind. pvalues
    rowData = df.iloc[ i, 2: ]
    rowData2 = df2.iloc[ i, 2: ]
    ind_corr_p_val2.append(statsmodels.stats.multitest.multipletests(ind_corr_pValue[i], alpha=0.05, method='fdr_tsbky', is_sorted=False, returnsorted=False))

    ind_corr_pval_reject.append(ind_corr_p_val2[i][0][0])  # we  get  whether to regect null hypothesis "True" or accept "False" and store it in an empty arrary called ind_corr_pval_reject

df['reject_null_hypothesis_ind_corrected'] = ind_corr_pval_reject # we add new column with reject_null_hypothesis_ind_corrected
Writer=pd.ExcelWriter(r'./TRYERDATA225.xlsx')
df.to_excel(Writer,'new_Sheet')
Writer.save()

paired = []
paired_pval = []
for i in range(0,df.shape[0]): # this for Loop Is To Get Pvalues of paired samples
    rowData = df.iloc[ i, 2:50 ]
    rowData2 = df2.iloc[ i, 2:50 ]
    paired.append(stats.ttest_rel((rowData.to_numpy()).astype(float),(rowData2.to_numpy()).astype(float))) # the method ttest_ind returns two value 1st :statistics 2nd : pvalue so we Sort it in an empty array called  paired
    paired_pval.append(paired[i][1]) # we get only pvalue from array ind  and append it in an ind_pval array


df['paired_pval'] = paired_pval # we add new column in excel with paired_pval
Writer=pd.ExcelWriter(r'./TRYERDATA225.xlsx')
df.to_excel(Writer,'new_Sheet')
Writer.save()

print ('=' * 150)
paired_corr = []
paired_corr_pValue = []
paired_reject = []
for i in range(0,df.shape[0]): # we get from this for loop the corrected pvalues of paired samples and whether to regect or accept null hypothesis
    rowData = df.iloc[ i, 2: ]
    rowData2 = df2.iloc[ i, 2: ]
    paired_corr.append(statsmodels.stats.multitest.multipletests(paired_pval[i], alpha=0.05, method='fdr_tsbky', is_sorted=False, returnsorted=False))

    paired_corr_pValue.append(paired_corr[i][1][0]) # we  get  the corrected pvalue and store it in an empty arrary called paired_corr_pValue
    paired_reject.append(paired_corr[i][0][0]) # we  get  whether to regect null hypothesis "True" or accept "False" and store it in an empty arrary called paired_reject

df['paired_pval_corr'] = paired_corr_pValue # we add new column in excel with paired_pval_corr
Writer=pd.ExcelWriter(r'./TRYERDATA225.xlsx')
df.to_excel(Writer,'new_Sheet')
Writer.save()

df['reject_null_hypothesis_paired'] = paired_reject # we add new column in excel with reject_null_hypothesis_paired
Writer=pd.ExcelWriter(r'./NewTryer309.xlsx')
df.to_excel(Writer,'new_Sheet')
Writer.save()

paired_corr2 = []
paired_corr_pval_reject = []
for i in range(0,df.shape[0]): # we use this for loop to get only whether to regect null hypothesis or accept it after correction of ind. pvalues
    rowData = df.iloc[ i, 2: ]
    rowData2 = df2.iloc[ i, 2: ]
    paired_corr2.append(statsmodels.stats.multitest.multipletests(paired_corr_pValue[i], alpha=0.05, method='fdr_tsbky', is_sorted=False, returnsorted=False))
    paired_corr_pval_reject.append(paired_corr2[i][0][0]) # we  get  whether to regect null hypothesis "True" or accept "False" and store it in an empty arrary called paired_corr_pval_reject

df['reject_null_hypothesis_paired_corrected'] = paired_corr_pval_reject # we add new column in excel with reject_null_hypothesis_paired_corrected
Writer=pd.ExcelWriter(r'./TRYERDATA225.xlsx')
df.to_excel(Writer,'new_Sheet')
Writer.save()


num_true_ind_before = np.sum(ind_pval_reject) # we Get No. Of True Values of independant samples before fdr correction
num_true_ind_after = np.sum(ind_corr_pval_reject) # we Get No. Of True Values of independant samples after fdr correction

num_true_paired_before = np.sum(paired_reject) # we Get No. Of True Values of paired samples before fdr correction
num_true_paired_after = np.sum(paired_corr_pval_reject) # we Get No. Of True Values of paired samples after fdr correction

print ("No. Of True Values of independant samples before fdr correction:   %s" %num_true_ind_before )
print ( "No. Of True Values of independant samples after fdr correction:   %s" %num_true_ind_after )
print( "No. Of True Values of paired samples before fdr correction:   %s" %num_true_paired_before )
print ( "No. Of True Values of paired samples after fdr correction:   %s" %num_true_paired_after )

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