import numpy as np
import pandas as pd
from datetime import datetime
from difflib import SequenceMatcher
from sklearn.decomposition import PCA
df=pd.read_csv('G:/Lord of the machines/campaign_data.csv')
df1=pd.read_csv('G:/Lord of the machines/train.csv')
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
le=preprocessing.LabelEncoder()
df['communication_type']=le.fit_transform(df['communication_type'])
def similiar(a,b):
    return SequenceMatcher(None,a,b).ratio()


for i in range(0,52):
    for j in range(i+1,52):
        sim=similiar(str(df.ix[i,['subject']]),str(df.ix[j,['subject']]))
        if sim>0.7:
            df.ix[j,['subject']]=df.ix[i,['subject']]

for i in range(0,52):
    for j in range(i+1,52):
        sim=similiar(str(df.ix[i,['email_body']]),str(df.ix[j,['email_body']]))
        if sim>0.7:
            df.ix[j,['email_body']]=df.ix[i,['email_body']]



df['email_body']=le.fit_transform(df['email_body'])
df['subject']=le.fit_transform(df['subject'])

df3=pd.merge(df,df1,on='campaign_id',sort=False)
df3=df3.drop(['campaign_id','email_url','time'],axis=1)
df3.loc[(df3['date1'] > 0) & (df3['date1'] <= 8), 'date1'] = 0
df3.loc[(df3['date1'] > 8) & (df3['date1'] <= 16), 'date1'] = 1
df3.loc[(df3['date1'] > 16) & (df3['date1'] <= 24), 'date1'] = 2
df3.loc[(df3['date1'] > 24) & (df3['date1'] <= 32), 'date1'] = 3
pca=PCA(.95)

from sklearn.utils import shuffle
df3=shuffle(df3)
df3=pca.fit_transform(df3)
Y=df3['is_open']
Z=df3['is_click']
X=df3.ix[:,df3.columns!='is_click']
K=X
X=X.ix[:,X.columns!='is_open']
X=StandardScaler().fit_transform(X)
K=StandardScaler().fit_transform(K)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
K_train,K_test,Z_train,Z_test=train_test_split(K,Z,test_size=0.3)
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

clf=RandomForestClassifier(random_state=0)
model=clf.fit(X_train,Y_train)
pi=model.score(X_test,Y_test)
score1=cross_val_score(clf,X,Y,cv=10)
print(score1)

from sklearn.metrics import accuracy_score


clf1=RandomForestClassifier(random_state=0)
model2=clf1.fit(K_train,Z_train)
p=model2.score(K_test,Z_test)
score2=cross_val_score(clf1,K,Z,cv=10)
print(score2)

df_test=pd.read_csv('G:/Lord of the machines/test_BDIfz5B.csv')
df4=pd.merge(df,df_test,on='campaign_id',sort=False)
df4=shuffle(df4)
df4=df4.drop(['campaign_id','email_url','time','send_date','send_dater'],axis=1)
df4['is_open']=model.predict(df4)
df4['is_click']=model2.predict(df4)
print(df4)
df6=df4.ix[:,['id','is_click']]
df6.to_csv('G:/Lord of the machines/answer.csv')



