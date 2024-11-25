#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.datasets import load_iris
import numpy as np

# - cross_val_score 
# - 교차검증을 사용할 수 있는 여러 방법들 중 하나
# - cross_val_score : 교차검증 간단하게 수행할 수 있는 함수
#     - k-fold 교차검증을 사용하고, k는 분석가 지정할 수 있다.
#     - 클래스의 불균형 상관없이 알아서 잘 확인하고 하니 너무 편하게 사용할 수 있다.
# - cross_validate - 교차검증 실행할 수 있음 
#     - 여러가지 추가 정보들을 확인할 수 있다. 테스트 훈련시간이나, 테스트 시간 등, 정밀도 재현율, 다른 평가지표 한 번에 볼 수 있음
#     - 기능적으로 좀 더 세부적으로 볼 수 있다.

# In[2]:


# x,y 데이터로드

iris= load_iris()
X,y = iris.data, iris.target

# In[5]:


clf = RandomForestClassifier(random_state=111)

# In[7]:


#cross_val_score 사용해서 교차검증 진행
#cross_val_score(모델, x,y, cv= , scoring=?)
# train, test를 나누지 않았다.
# X, y만 넣어서 진행했다.
scores = cross_val_score(clf, X,y, cv=5, scoring='accuracy')

# In[8]:


scores

# In[10]:


scores.mean() #평균에 대한 정확도 나오는지 확인할 수 있다.

# In[11]:


#cross_validate 사용법
#cross_validate(모델, x,y, cv=?, scroing=?, return_train_score=True?) 
scoring=['accuracy','precision','recall','f1']

results= cross_validate(clf, X,y, cv=5, scoring=scoring, return_train_score=True)

# In[12]:


results

# - precision, recall, f1 nan 값이 나왔다.
# - why? 이런 값이 나오는가?
# - 이진분류와 다중분류에 따른 값의 계산이 다르다.
# - 대부분 알고 있는 평가지표는 이진분류가 디폴트 
# - iris 다중분류가 된다. 이진분류로 생각하는 평가지표에서 nan 
# 
# ### 다중분류의 precision, recall, f1스코어는 계산 방법
# - macro
# - micro
# - weight 
# 
# - 다중분류의 평가를 계산할 수 있다.
# 

# In[14]:


scoring=['accuracy','precision_macro','recall_macro','f1_macro']

results_mc= cross_validate(clf, X,y, cv=5, scoring=scoring, return_train_score=True)

# In[15]:


results_mc

# In[16]:


scoring=['accuracy','precision_micro','recall_micro','f1_micro']

results_mic= cross_validate(clf, X,y, cv=5, scoring=scoring, return_train_score=True)

# In[17]:


results_mic

# - 기본적으로 binary 디폴트로 해서 클래스 재현율, 정밀도 등을 계산한다.
# 
# ### - 다중 클래스 분류문제의 경우
# - micro 
#     - 가중치간의 weight는 무시하고 전체 샘플 수로 계산한다. 모든 클래스가 동등하게 중요한 경우 사용
# - macro 
#     - 클래스간의 불균형이 큰 경우에 사용한다.
# - weighted 
#     - 클래스간의 불균형인 경우 가중치를 어디에 더 두는 것에 따라서 다르게 계산되는 것

# ### precision
# - Class A : 1 TP & 1 FP = 0.5
# - Class B : 10 TP & 90 FP = 0.1
# - Class C : 1TP & 1FP = 0.5
# - Class D : 1TP & 1FP = 0.5
# 
# - precision을 만든다 하면? 
# - Macro -Precision : 평균의 평균을 낸다. ( 0.5+0.1+0.5+0.5 )/4  =0.4
# 
# 
# - Micro- Precision : 기존 평균을 내는 방식돠 동일하게 진행  (1+10+1+1)/(2+100+2+2)= 0.123
# 
# ----
# - 클래스의 불균형, 다중분류 등일 때 평가지표를 어떤 것으로 사용해야 하는가?
# - 이런 방식이 다르기 때문에 도메인에 맞게 사용하면 된다.
# - 클래스의 불균형이 있는 데이터셋이나 케이스의 경우에는 Micro 좀더 효과적인 평가지표

# ### 이진분류시 어떻게 되는지

# In[50]:


import seaborn as sns
df = sns.load_dataset('titanic')

# In[47]:


df['survived'].value_counts()

# In[51]:


df_tt=df[['survived','pclass','fare','age']]

# In[52]:


df_tt=df_tt.dropna()

# In[53]:


df_tt_x= df_tt[['pclass','fare','age']]
df_tt_y= df_tt['survived']

# In[54]:


cross_val_score(clf, df_tt_x,df_tt_y, cv=5, scoring='accuracy') #na값이 있는 경우는 에러 

# In[55]:


scoring=['accuracy','precision','recall','f1']

# In[56]:


results= cross_validate(clf, df_tt_x,df_tt_y, cv=5, scoring=scoring, return_train_score=True)

# In[57]:


results

# - 세 개 이상의 모델을 cross_val_score로 계산하기

# In[58]:


model=[DecisionTreeClassifier(),RandomForestClassifier(),LogisticRegression()]
name = ['DT','RF','LR']

for model, name in zip(model,name):
    print("### 사용할 알고리즘",name,'###')
    for score in ['accuracy','precision','recall','f1']:
        print(score)
        print('-----')
        print(cross_val_score(model,df_tt_x, df_tt_y, scoring=score, cv=5))

# In[61]:


#추가 교차검증 내용 LeaveOneOut
#시간이 꽤 오래 걸린다.

from sklearn.model_selection import LeaveOneOut
clf = RandomForestClassifier(random_state=111)

loo = LeaveOneOut()

scores = cross_val_score(clf, df_tt_x, df_tt_y, cv=loo) #cv를 LeaveOneOut 넣어서 진행하면

# ## ML평가지표들
# 
# - Confusion Matrix 
# 
# 
#         - 예측 클래스 
#         - N(0) / P(1)
#     
#     - N(0)  - TN  FP
# - 실제클래스
#     - P(1)  - FN  TP

# ### 정확도 
# - 예측 결과 동일한 데이터 개수/ 전체 예측 데이터 개수 
# - 정확도 계산식
# - (TN + TP)/ (TN+TP+FN+FP)
# - 이 계산을 통해서 정확도가 나오는 것
# 
# - 정확도만 보면 안 되는 이유?
# - 정확도만 보면 안 되는 이유 -> TN , TP 클래스 불균형 이슈가 생기면 예측하는 게 당연한 결 예측하고 정확도가 높게 나온다.
# - 사기탐지 10개 사기, 1000개 정상, 정상 찾는것도 정확도 계산되니깐 문제가 발생한다.
#     

# ### 정밀도
# 
# - TP / (FP+TP)
# - 1로 예측한 것들중에서 진짜 1인 경우가 얼마나 되는가?

# ### 재현율
# - TP/(FN+TP)
# - 실제 값이 P 대상 중에서 예측과 실제 값이 P로 일치한 비율 
# - 민감도라고도 한다.

# - 정밀도와 재현율은 Trade off 관계이다.
# 
# - F1스코어
# - AIC, ROC커브 등

# ### 임계값에 따라서 모든 지표의 값이 다 달라질 수 있다.
# - 통상적으로 생각하는 것은 임계값이 0.5 이진분류시 0.5 기준으로 0.5보다 초과면 1, 0.5미만이면 0 이런 식으로 이진분류로 들어간다.
# - 도메인, 상황이나 따라서 임계값을 변경해야 하는 경우가 있다. 0.6 0.4 , 0.7 ,0.3 값들이 달라지게 된다. 0.3, 0.7 값이 완전히 달라진다.

# In[ ]:



