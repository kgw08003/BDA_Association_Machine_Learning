#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt

# 필요한 패키지 설치

import pandas as pd 
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import *
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[6]:


#타이타닉 데이터 추출
df=sns.load_dataset('titanic').dropna(subset=['age','embarked'])

# In[8]:


df['sex']= df['sex'].map({'male':0, 'female':1})
df=pd.get_dummies(df, columns=['embarked'],drop_first=True)

# In[13]:


#피처 추출

df.columns

# In[14]:


X = df[['sex','age','fare','embarked_Q','embarked_S']]
y = df['survived']

# ### 로지스틱을 코드로 구현

# In[17]:


#데이터 분할
X_train, X_test,y_train,y_test= train_test_split(X,y, test_size=0.2, random_state=111)

# In[21]:


#스케일링 진행
scaler= StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



# In[30]:


# 넘파이 배열로 변환 
y_train = y_train.values
y_test  = y_test.values

# In[35]:


## 시그모이드 함수
def sigmoid(z):
    return 1 / (1+np.exp(-z))

## 비용함수 정의 : 로지스틱 회귀의 손실함수
## X : 피처,y : 예측값, w : 가중치, b : 절편
def compute_cost(X,y,w,b):
    m =len(y) #데이터 전체 포인트 수
    cost = 0
    for i in range(m):
        z = np.dot(X[i],w)+b#선형의 조합
        h = sigmoid(z) #예측 확률
        cost += -y[i] * np.log(h) - (1 - y[i]) * np.log(1 - h)
    cost = cost/m #평균비용으로 계산
    return cost

#초기 가중치와 절편을 최적화
w = np.zeros(X_train.shape[1]) # 0으로 초기화
b = 0 #0으로 초기화

## 경사 하강법을 최적화 함수 정의
## X : 피처,y : 예측값, w : 가중치, b : 절편 , learning_rate, num_iterations
## gradient_descent -> w,b, 최적을 찾는 것 return 값은 w,b 반환값 
def gradient_descent(X, y, w, b, learning_rate, num_iterations):
    m = len(y)
    for _ in range(num_iterations):
        dw = np.zeros(X.shape[1]) #가중치 그레디언트 0 초기화
        db = 0 #가중치의 절편 0 초기화
        for i in range(m):
            z = np.dot(X[i], w)+b # 선형 조합
            h = sigmoid(z) #예측확률
            #가중치를 계산
            dw += (h - y[i]) * X[i] #가중치 계산
            db += (h - y[i]) #절편 계산
        dw = dw/m #평균에 대한 그레디언트
        db = db/m #평균에 대한 그레디언트
        w = w - learning_rate * dw #가중치 업데이트
        b = b - learning_rate * db #절편 업데이트
    return w, b

# 하이퍼파라미터 설정
learning_rate = 0.01 # 학습률
num_iterations = 1000 # 반복 횟수

## 경사하강법으로 최적의 가중치와 절편을 찾아야 한다.
w_opt, b_opt = gradient_descent(X_train, y_train, w, b, learning_rate, num_iterations)

# 모델 평가 
def predict(X, w, b):
    m = X.shape[0] # 데이터의 전체 수
    y_pred = np.zeros(m) #0값으로 배열 만들기
    for i in range(m):
        z = np.dot(X[i],w) +b
        h = sigmoid(z)
        y_pred[i] = 1 if h >= 0.5 else 0 # 임계값 기준으로 0.5 기준 1로 바라보고 , 0.49 0으로 보는 것 # 임계값은 조정이 가능하다.
    return y_pred


#타이타닉 생존예측값 출력
y_pred_lr =predict(X_test, w_opt, b_opt)

# In[39]:


#예측값과 실제 값을 비교함
accuracy = np.mean(y_pred_lr == y_test)

# In[40]:


print(accuracy)

# In[36]:


y_pred_lr

# In[38]:


y_test

# In[ ]:



