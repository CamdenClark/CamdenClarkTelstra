
# coding: utf-8

# In[2]:

import numpy as np
import pandas as pd
import matplotlib


# In[28]:

test=pd.read_csv("test.csv") 
testUniqueIds=pd.unique(test.id.ravel())
train=pd.read_csv("train.csv")
log_feature=pd.read_csv("log_feature.csv") #TODO

def typify(self):
    if self==each:
        return 1
    return 0

thetypes=log_feature.log_feature.unique()
length_of_features=len(log_feature.index)
    
for each in thetypes:
    log_feature[each]=pd.Series(np.zeros(length_of_features,dtype=np.int))
    log_feature[each]=log_feature.log_feature.apply(typify)
    log_feature[each]=log_feature[each]*log_feature['volume']
#log_feature=log_feature.drop(['volume'],axis=1)
#log_feature=log_feature.drop([''])
log_feature=log_feature.groupby("id").sum()
log_feature.index.name=None
    
train.set_index("id",drop=True,inplace=True)
train.index.name=None
#del test['location']
#del train['location']
train['location']=train['location'].apply((lambda x: int(x[9:])))
    
test.set_index("id",drop=True,inplace=True)
test.index.name=None
test['location']=test['location'].apply((lambda x: int(x[9:])))


event_type=pd.read_csv("event_type.csv")
thetypes=event_type.event_type.unique()
length_of_events=len(event_type.index)

    
for each in thetypes:
    event_type[each]=pd.Series(np.zeros(length_of_events,dtype=np.int))
    event_type[each]=event_type['event_type'].apply(typify)
event_type=event_type.groupby("id").sum()
event_type.index.name=None

resource_type=pd.read_csv("resource_type.csv")
thetypes=resource_type.resource_type.unique()
length_of_resources=len(resource_type.index)
for each in thetypes:
    resource_type[each]=pd.Series(np.zeros(length_of_resources,dtype=np.int))
    resource_type[each]=resource_type['resource_type'].apply(typify)
resource_type=resource_type.groupby("id").sum()
resource_type.index.name=None

severity_type=pd.read_csv("severity_type.csv")

severity_type.set_index("id",drop=True,inplace=True)
severity_type.index.name=None
severity_type['severity_type']=severity_type['severity_type'].apply((lambda x: int(x[14:])))
#trainresult=pd.concat([train,severity_type,resource_type,event_type],axis=1,join='inner')
#testresult=pd.concat([test,severity_type,resource_type,event_type],axis=1,join='inner')
trainresult=pd.concat([train,severity_type,event_type,resource_type,log_feature],axis=1,join='inner')
testresult=pd.concat([test,severity_type,event_type,resource_type,log_feature],axis=1,join='inner')
drops=['feature 210','feature 330']
trainresult=trainresult.drop(drops,axis=1)
testresult=testresult.drop(drops,axis=1)


# In[29]:

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.neural_network import MLPClassifier
x_train=trainresult.drop(["fault_severity"],axis=1)
y_train=trainresult["fault_severity"]
x_test =testresult.copy()
#model=SVC()
model=LogisticRegression()

#model=RandomForestClassifier(min_samples_split=5,min_samples_leaf=2,n_estimators=100)
#model=AdaBoostClassifier(RandomForestClassifier(),n_estimators=100)
model.fit(x_train,y_train)
print(model.score(x_train,y_train))
y_probs=model.predict_proba(x_test)
print(y_probs)
final=[[iden,(y_probs[j][0]),(y_probs[j][1]),(y_probs[j][2])] for j,iden in enumerate(testUniqueIds)]
frame=pd.DataFrame(final,columns=('id','predict_0','predict_1','predict_2'))
frame.to_csv("submission.csv",index=False)


# In[24]:

from sklearn.cross_validation import cross_val_score
scores = cross_val_score(model, x_train, y_train, cv=20, scoring='accuracy')
print(scores)


# In[32]:

import xgboost as xgb
x_train_xgb=x_train.as_matrix()
y_train_xgb=y_train.as_matrix()
model=xgb.XGBClassifier()
from sklearn.cross_validation import cross_val_score
scores = cross_val_score(model, x_train_xgb, y_train_xgb, cv=5, scoring='log_loss')


# In[33]:

print(scores.mean())


# In[34]:

model.fit(x_train_xgb,y_train_xgb)
print(model.predict_proba(x_test.as_matrix()))


# In[22]:

x_train.columns


# In[25]:

x_train['feature 210'].value_counts()


# In[36]:

import xgboost as xgb
x_train_xgb=x_train.as_matrix()
y_train_xgb=y_train.as_matrix()
model=xgb.XGBClassifier()
model.fit(x_train_xgb,y_train_xgb)
y_probs=model.predict_proba(x_test.as_matrix())
print(y_probs)
final=[[iden,(y_probs[j][0]),(y_probs[j][1]),(y_probs[j][2])] for j,iden in enumerate(testUniqueIds)]
frame=pd.DataFrame(final,columns=('id','predict_0','predict_1','predict_2'))
frame.to_csv("submission.csv",index=False)


# In[ ]:



