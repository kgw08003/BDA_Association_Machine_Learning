#!/usr/bin/env python
# coding: utf-8

# In[20]:


from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score #Confusion matrix 수업 때 진행할 예정 
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score#임포트 
import numpy as np
import pandas as pd

# - kfold의 문제점을 간단히 살펴보고 코드로
# - 다양한 교차검증 코드를 진행할 예정

# In[24]:


fold_iris = load_iris()
features = fold_iris.data

label = fold_iris.target

# In[25]:


features

# In[26]:


label

# - 교차검증을 통해서 나누기

# In[39]:


from sklearn.model_selection import KFold

kfold = KFold(n_splits=5)

cv_acc_train=[]
cv_acc_test=[]
kf_ml = DecisionTreeClassifier(random_state=111,max_depth=3)

# In[36]:


kfold

# In[40]:


n_iter = 0

for train_idx, test_idx in kfold.split(features):
    X_train, X_test = features[train_idx], features[test_idx]
    y_train, y_test = label[train_idx], label[test_idx]
    
    #dt모델 학습하기
    kf_ml.fit(X_train, y_train)
    
    #예측
    kf_pred_train =kf_ml.predict(X_train)
    kf_pred_test =kf_ml.predict(X_test)
    
    # 정확도를 5번 측정할 것
    
    n_iter +=1
    acc_train = np.round(accuracy_score(y_train, kf_pred_train),4)
    acc_test = np.round(accuracy_score(y_test, kf_pred_test),4)
    
    #교차검증 train, test 정확도 확인
    print('\n {} 번 train 교차 검증 정확도 :{}, test의 교차검증 정확도 :{}'.format(n_iter, acc_train, acc_test))
    
    cv_acc_train.append(acc_train)
    cv_acc_test.append(acc_test)
    

print('train 평균 정확도', np.mean(cv_acc_train))
print('test 평균 정확도', np.mean(cv_acc_test))

# - skf 모델을 통해서 kfold 문제 해결

# In[41]:


from sklearn.model_selection import StratifiedKFold

# In[46]:


skf_iris = StratifiedKFold(n_splits=5)
cnt_iter = 0

# In[47]:


skf_iris

# In[48]:


n_iter = 0

skf_cv_acc_train=[]
skf_cv_acc_test=[]
skf_ml = DecisionTreeClassifier(random_state=111,max_depth=3)

#skf 사용한 교차검증 
for train_idx, test_idx in skf_iris.split(features,label): #skf split 안에 label 
    X_train, X_test = features[train_idx], features[test_idx]
    y_train, y_test = label[train_idx], label[test_idx]
    
    #skf_dt모델 학습하기
    skf_ml.fit(X_train, y_train)
    
    #예측 (skf, split을 통해 진행)
    skf_pred_train =skf_ml.predict(X_train)
    skf_pred_test =skf_ml.predict(X_test)
    
    # 정확도를 5번 측정할 것
    
    n_iter +=1
    acc_train = np.round(accuracy_score(y_train, skf_pred_train),4)
    acc_test = np.round(accuracy_score(y_test, skf_pred_test),4)
    
    #교차검증 train, test 정확도 확인
    print('\n {} 번 train 교차 검증 정확도 :{}, test의 교차검증 정확도 :{}'.format(n_iter, acc_train, acc_test))
    
    skf_cv_acc_train.append(acc_train)
    skf_cv_acc_test.append(acc_test)
    

print('train 평균 정확도', np.mean(skf_cv_acc_train))
print('test 평균 정확도', np.mean(skf_cv_acc_test))

# In[50]:


len(features)

# In[53]:


import pandas as pd
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns = iris.feature_names)
iris_df['label'] = iris.target

# In[54]:


iris_df

# In[56]:


kfold = KFold(n_splits=5)
n_iter = 0

for train_idx, test_idx in kfold.split(iris_df):
    n_iter +=1
    lb_train = iris_df['label'].iloc[train_idx]
    lb_test = iris_df['label'].iloc[test_idx]
    print('학습 정답 레이블', lb_train.value_counts())
    print('테스트 정답 레이블', lb_test.value_counts())

# In[57]:


skfold = StratifiedKFold(n_splits=5)
n_iter = 0

for train_idx, test_idx in skfold.split(iris_df, iris_df['label']):
    n_iter +=1
    lb_train = iris_df['label'].iloc[train_idx]
    lb_test = iris_df['label'].iloc[test_idx]
    print('학습 정답 레이블', lb_train.value_counts())
    print('테스트 정답 레이블', lb_test.value_counts())

# In[ ]:



