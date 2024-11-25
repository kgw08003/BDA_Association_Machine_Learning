#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# In[4]:


# iris data 불러오기
iris = load_iris()
iris_data =iris.data
iris_label = iris.target
iris_df = pd.DataFrame(data=iris.data, columns = iris.feature_names)
## 데이터셋 정의

iris_df['y']= iris_label

# In[7]:


iris_df.y.value_counts() # 클래스 다중분류, 균등하게 50개씩

# ## train_test_split을 통해 데이터를 나눈다.

# In[10]:


#데이터 분리
from sklearn.model_selection import train_test_split

#train_test 데이터셋 분리
X_train, X_test, y_train, y_test = train_test_split(iris_data, iris_label, test_size=0.3, random_state=111)

# - test_size : train,test 나누는 비중 
# - random_state : 카드게임 포커 ( 포커 자체를 섞는다 ) 1번 섞고 2번 섞고 (처음 시작하는 포커의 순서는 동일,  섞는 것도 동일하게 섞는다 )
# - random_seed의 개념과 같은 난수 지정 

# In[15]:


y_test

# In[17]:


y_train

# ## DT를 통해서 모델에 학습

# In[20]:


#모델링은 불러오기만 하면 된다.
df_clf = DecisionTreeClassifier(random_state=111)
# dt 모델을 불러옴

# In[21]:


df_clf

# ### 모델을 불러왔으니 ->학습을 하면 된다.
# - train 데이터로 학습하고 -> test 데이터로 검증한다.

# - 학습시킬 때 사용하는 함수는 fit

# In[22]:


df_clf.fit(X_train, y_train) # 모델에 train 데이터를 학습한다.

# - test를 검증해야 한다.
# - test와 어떤 걸 가지고 검증해야 할까?
# - 머신에게 예측을 해봐! 예측값을 가지고 실제값과 예측값의 차이로 비교하는 것, 성능 평가
# 
# - X_train, y_train -> 학습을 위해 사용하는 데이터 셋
# - X_test, y_test -> 검증을 위해 사용하는 데이터셋
# - y_test , ????? -> 누구랑 비교를 할 것인가?
# --- 
# - ???? -> 머신이 train 예측한 값을 y_test 비교
# --- 
# - y_test, 예측값을 머신을 통해 출력(y_pred) -> 정확도, 정밀도, 재현율, F1스코어, Roc 등등 비교를 하는 것

# - 예측을 위해서 predict  예측값을 출력 할 예정

# In[25]:


train_pred =df_clf.predict(X_train)

# In[26]:


train_pred

# In[28]:


from sklearn.metrics import accuracy_score #사이킷런패키지에서 불러오는 정확도 평가 메트릭스

print('DT train 정확도 : {0:.4f}'.format(accuracy_score(y_train, train_pred))) # accuracy_score(실제값, 예측값 )

# - train 100% 나옴
# - 과적합 발생

# - test를 가지고 실제 진행

# In[29]:


test_pred = df_clf.predict(X_test) # test 데이터로 실제 값을 예측

# In[30]:


test_pred

# In[31]:


print('DT train 정확도 : {0:.4f}'.format(accuracy_score(y_train, train_pred))) # accuracy_score(실제값, 예측값 )
print('DT test 정확도 : {0:.4f}'.format(accuracy_score(y_test, test_pred))) # accuracy_score(실제값, 예측값 )

# - train 100% , test 93% 
# - base 모델

# - 100% train의 과적합 -> 문제 발생. train 너무 핏한 모델
# - 다양한 방법으로 해당 내용들을 과적합 되지 않게 진행해야 한다.

# ## Dt의 하이퍼파라미터를  조정하여 오버피팅을 막기

# In[34]:


df_clf1 = DecisionTreeClassifier(max_depth =2 ,random_state=111)

# In[36]:


df_clf1.fit(X_train, y_train)

# In[37]:


train_pred1 =df_clf1.predict(X_train)

# In[38]:


print('DT train 정확도 : {0:.4f}'.format(accuracy_score(y_train, train_pred1))) # accuracy_score(실제값, 예측값 )

# - 과적합을 막기위해서 -> 모델링에 하이퍼파라미터를 튜닝하게 되어서 100% -> 97% 정도로 잡았다.
# - 모델링에 대한 일반화를 하기 위해서 이런 작업을 하는 것

# In[ ]:



