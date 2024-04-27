import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
import pickle

warnings.filterwarnings("ignore")

df = pd.read_csv('HepatitisCdata.csv')

df.info()

df.isnull().sum()

ALB_mean = df['ALB'].mean()
df['ALB'].fillna(ALB_mean,inplace = True)

ALP_mean = df['ALP'].mean()
df['ALP'].fillna(ALP_mean,inplace = True)

ALT_mean = df['ALT'].mean()
df['ALT'].fillna(ALT_mean,inplace = True)

CHOL_mean = df['CHOL'].mean()
df['CHOL'].fillna(CHOL_mean,inplace = True)

PROT_mean = df['PROT'].mean()
df['PROT'].fillna(PROT_mean,inplace = True)

df.info()

df.columns

df.drop(['Unnamed: 0'], axis=1,inplace=True)

df.columns

df.columns = ['Category', 'Age', 'Sex','ALB: g/dL','ALP: IU/L','ALT: IU/L','AST: IU/L',
              'BIL: mg/dL','CHE: U/L','CHOL: mg/dL','CREA: mg/dL','GGT: IU/L','PROT: g/dL']

df.info()

numerical_columns = ['Age','ALB: g/dL','ALP: IU/L','ALT: IU/L','AST: IU/L',
              'BIL: mg/dL','CHE: U/L','CHOL: mg/dL','CREA: mg/dL','GGT: IU/L','PROT: g/dL']

for column in numerical_columns:
    col_min = df[column].min()
    col_max = df[column].max()
    print(f"Smallest value in {column}: {col_min}")
    print(f"Largest value in {column}: {col_max}")
    print()

plt.boxplot(df['CREA: mg/dL'])
plt.title('Boxplot of CREA')
plt.ylabel('CREA')
plt.show()

df = df.drop(df[df['CREA: mg/dL'] > 400].index)

plt.boxplot(df['ALB: g/dL'])
plt.title('Boxplot of ALB')
plt.ylabel('ALB')
plt.show()

df = df.drop(df[df['ALB: g/dL'] > 70].index)

plt.boxplot(df['ALP: IU/L'])
plt.title('Boxplot of ALP')
plt.ylabel('ALP')
plt.show()

df = df.drop(df[df['ALP: IU/L'] > 200].index)

plt.boxplot(df['ALT: IU/L'])
plt.title('Boxplot of ALT')
plt.ylabel('ALB')
plt.show()

df = df.drop(df[df['ALT: IU/L'] > 200].index)

plt.boxplot(df['GGT: IU/L'])
plt.title('Boxplot of GGT')
plt.ylabel('GGT')
plt.show()

df = df.drop(df[df['GGT: IU/L'] > 200].index)

df['Category'].value_counts()

df['Category'] = df['Category'].replace({'0s=suspect Blood Donor' : 0, '0=Blood Donor' : 0, '1=Hepatitis':1,
                                         '3=Cirrhosis':1,'2=Fibrosis':1})

df['Sex'] = df['Sex'].replace({'m':1,'f':0})

df.info()

df.head()

X = df.iloc[:,1:]
Y = df.iloc[:,0]

X.head()

Y.head()

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=0.3,random_state=9)

from sklearn.preprocessing import StandardScaler
st_x= StandardScaler()
X_train= st_x.fit_transform(X_train)
X_test= st_x.transform(X_test)

from sklearn.tree import DecisionTreeClassifier
classifier= DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train, Y_train)

pickle.dump(classifier,open('DT.pkl','wb'))
classifier = pickle.load(open('DT.pkl','rb'))

