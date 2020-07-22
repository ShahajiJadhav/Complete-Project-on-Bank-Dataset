# Business Objective:To predict the Whether the customer will fall under default or not.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

Bank_Data = pd.read_csv('F:/Project/Project.csv')
Bank_Data.shape

dollor = ['DisbursementGross', 'BalanceGross', 'ChgOffPrinGr', 'GrAppv', 'SBA_Appv']
for i in dollor:
    for j in range(len(Bank_Data)):
        Bank_Data[i][j] = int(float(Bank_Data[i][j].replace('$', '').replace(',','')))        

#Removing Duplicates
df = pd.DataFrame(Bank_Data)
#Basic Info
df.info()
df.nunique()
df = df.drop_duplicates()
##Droping non important columns
df = df.drop(['Name', 'City', 'Bank', 'Zip','CCSC'], axis=1)
df.columns
# dropping features having missing values more than 60%
# df = df.drop((percent[percent > 0.6]).index,axis= 1)

##Since we Know that output variable is MIS Status. We will make column for default on the basis of ChgOffPrinGr
#OUTPUT VARIABLE ---> MIS_STATUS
sns.countplot(x='MIS_Status',data=df)
df['MIS_Status'].value_counts()
df['MIS_Status'].isna().sum()#868 NA count
len(df.loc[(bank.ChgOffPrinGr>0) & (df.MIS_Status=='P I F')]) #1319
len(df.loc[(df.ChgOffPrinGr==0) & (df.MIS_Status=='CHGOFF')]) #164

x=[] # store the row id of records having MIS_Status as NA
for i in range(len(df)):
    if df['MIS_Status'][i] not in ['CHGOFF','P I F']:
        x.append(i)
                
a=df.iloc[x,[18,22,23]]#analysing the related variables
#IF ChgOffPrinGr > 0 then it would be a defaulter

#Replacing the NA values of MIS_Status by comparing with values in ChgOffPrinGr
for i in range(len(bank)):
    if bank['MIS_Status'][i] not in ['CHGOFF','P I F']:
        if bank['ChgOffPrinGr'][i]>0:
            bank['MIS_Status'][i]='CHGOFF'
        else:
            bank['MIS_Status'][i]='P I F'

#Null values imputed

#creating output variable
bank['default']=bank['MIS_Status'].map({'P I F':0,'CHGOFF':1})








cat_col = df.select_dtypes(include = ['object']).columns ; cat_col
num_col = df.select_dtypes(exclude = ['object']).columns ; num_col
#We exclude dates 
cat_col = ['State', 'BankState', 'RevLineCr', 'LowDoc',  'MIS_Status', 'FranchiseCode', 'UrbanRural']
num_col =['Term', 'NoEmp', 'NewExist', 'CreateJob', 'RetainedJob',  'DisbursementGross', 'BalanceGross','ChgOffPrinGr', 'GrAppv', 'SBA_Appv']

#Reordering Columns
df = pd.DataFrame(df.loc[:,('Term', 'NoEmp', 'NewExist', 'CreateJob', 'RetainedJob','FranchiseCode', 'UrbanRural', 'DisbursementGross', 'BalanceGross',
      'ChgOffPrinGr', 'GrAppv', 'SBA_Appv', 'BankState','ApprovalDate','ApprovalFY','ChgOffDate', 'DisbursementDate','RevLineCr', 'LowDoc','MIS_Status')])




  
##Plotting Largest values from Dataframe    
for x in df.columns:
    print(df[x].value_counts().nlargest(15))
    df[x].value_counts().nlargest(10).plot(kind='bar', figsize=(16,9))
    plt.title('Frequency Distribution of %s' % x, fontsize = 15)
    plt.ylabel('Number of Occurrences of %s'%x, fontsize=15)
    plt.xlabel('%s'%x, fontsize=15)
    plt.show()

outliers = pd.DataFrame(columns=['Feature','Number of Outliers']) # Creating a new df to
for column in numeric_col: # Iterating thorough each feature            
        q1 = df[column].quantile(0.25) 
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        fence_low = q1 - (1.5*iqr)
        fence_high = q3 + (1.5*iqr)
        total_outlier = df[(df[column] < fence_low) | (df[column] > fence_high)].shape[0]
        outliers = outliers.append({'Feature':column,'Number of Outliers':total_outlier},ignore_index=True)

outliers


df['RevLineCr'].value_counts()
df['RevLineCr'] = df['RevLineCr'].replace({'0': "U", "1": "Y", "`": 'U', ",": "T" })
df['RevLineCr'] = df['RevLineCr'].fillna(df['RevLineCr'].value_counts().index[0])

df['LowDoc'].value_counts()
df['LowDoc'] = df['LowDoc'].replace('1', 'C')

#df.DisbursementDate.value_counts()#225 Null
#df['DisbursementDate'] = df['DisbursementDate'].fillna(df['DisbursementDate'].value_counts().index[0])

#df.MIS_Status.value_counts()#868 Null
#df['MIS_Status'] = df['MIS_Status'].fillna(df['MIS_Status'].value_counts().index[0])
#df['DisbursementGross'].equals(df['GrAppv'])


cols = ['Defaulted']  + [col for col in df if col != 'Defaulted']
df = df[cols] 


df['DisbursementYear']=0  #extracting the year alone from date
for i in range(len(df)):
    try:
        df['DisbursementYear'][i]=datetime.strptime(df['DisbursementDate'][i], "%d-%b-%y").year
        if i%5000==0:
            print(i)
    except:                    # due to presence of NA values
        pass

df['DisbursementYear'].value_counts()

df['GrAppv_Disbursement']=0   # 1 when GrAppv is less, 2 when GrAppv is more, 0 when equal
for i in range(len(df)):
    if df.GrAppv[i] < df.DisbursementGross[i]:
        df.GrAppv_Disbursement[i]=1
    elif df.GrAppv[i] > df.DisbursementGross[i]:
        df.GrAppv_Disbursement[i]=2



###Treating with missing values
df.isnull().sum()




## Plotting Scattergram to ckeck data distribution.
for i in df[num_col]:
    sns.


#### Making Dummy Variable for Categorical Columns

todummy_list =['MIS_Status','RevLineCr','LowDoc' ]

for x in todummy_list:
   print(df[x].value_counts())

def dummy_df(df, todummy_list):
    for x in todummy_list:
        dummies = pd.get_dummies(df[x], prefix=x, dummy_na=False)
        df = df.drop(x, 1)
        df = pd.concat([df, dummies], axis=1)
    return df

df = dummy_df(df, todummy_list)






#### EDA Completed

df = pd.DataFrame(df.loc[:,('Defaulted','Term', 'NoEmp', 'NewExist', 'CreateJob', 'RetainedJob','FranchiseCode', 'UrbanRural', 'DisbursementGross', 'BalanceGross',
       'ChgOffPrinGr', 'GrAppv', 'SBA_Appv','Bank', 'BankState','ApprovalDate','ApprovalFY','ChgOffDate', 'DisbursementDate','RevLineCr', 'LowDoc','MIS_Status')])



# Checking For imbalancieng
# we are finding the percentage of each class in the feature 'y'
class_values = (df['Defaulted'].value_counts()/df['Defaulted'].value_counts().sum())*100
print(class_values)

##Checking For Outliers in continuous data
for i in num_col

out = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]


list = ['MIS_Status','LowDoc']
target_var = 'default'
for i in list:
    print(pd.crosstab(df[target_var],df[i]))
    print(df[[i,target_var]].groupby([i]).mean().sort_values(by= target_var, ascending=False))


from scipy.cluster.vq import kmeans
centroids, avg_distance = kmeans(df, 4)
groups, cdist = cluster.vq.vq(df, centroids)
plt.scatter(df, np.arange(0,100), c=groups)
plt.xlabel('df_xyz')
plt.ylabel('Indices')
plt.show()




top_corr_features = df.corr().index
plt.figure(figsize = (16,9))
sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")

from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
# fit an Extra Trees model to the data
model = ExtraTreesClassifier(n_estimators=18)
model.fit(df.iloc[:, 15:], df.iloc[:,1]) 
# display the relative importance of each attribute
print(model.feature_importances_)
feat_importances = pd.Series(model.feature_importances_, index=df.iloc[:,8:].columns)
feat_importances.nlargest(18).plot(kind='barh')
plt.show()

?FranchiseCode,

df.MIS_Status.value_counts()
df.ChgOffPrinGr.value_counts()
df.ChgOffDate.isna().sum()
df.ChgOffDate.value_counts()
RiskAmount = df.DisbursementGross - df.SBA_Appv
amt = []
for R_Amt in RiskAmount:
    if R_Amt >0:
       amt.append(R_Amt)
    else:
        amt.append(0)



df['RiskAmount'] = amt
df.columns
df.groupby(["ApprovalFY","DisbursementDate"]).count()
##Describing months and year
LoanTime = df.loc[:, ('ApprovalDate', 'ApprovalFY', 'DisbursementDate','ChgOffDate')]
LoanTime['DisbursementDate']= pd.to_datetime(LoanTime['DisbursementDate'])
LoanTime['DisbursementMonth'] = LoanTime['DisbursementDate'].dt.month
LoanTime['DisbursementYear'] = LoanTime['DisbursementDate'].dt.year
LoanTime['ApprovalDate'] = pd.to_datetime(LoanTime['ApprovalDate'])
LoanTime['ApprovalMonth'] = LoanTime['ApprovalDate'].dt.month
LoanTime['ApprovalYear'] = LoanTime['ApprovalDate'].dt.year
LoanTime['Defaulted'] = df['Defaulted']
ltc = ['DisbursementMonth', 'DisbursementYear', 'ApprovalMonth','ApprovalYear', 'Defaulted']
#['ApprovalDate', 'ApprovalFY', 'DisbursementDate', 'ChgOffDate','DisbursementMonth', 'DisbursementYear', 'ApprovalMonth', 'ApprovalYear']

for y in LoanTime[ltc]:
    LoanTime[z].value_counts().nlargest(40).plot(kind = 'bar', figsize = (16, 9))
    plt.title('Top 40 %s'%y)
    plt.ylabel('Frequancy of %s'%y, fontsize = 11)
    plt.xlabel('%s'%y, fontsize = 11)
    plt.show()

plt.ylabel('Frequancy of %s'%z, fontsize = 11)
    plt.xlabel('Default Declaired', fontsize = 11)
    plt.show()










