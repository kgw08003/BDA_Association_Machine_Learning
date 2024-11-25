#!/usr/bin/env python
# coding: utf-8

# ### 주요 하이퍼파라미터
# 
# 1. **penalty**: 사용될 규제(regularization) 유형을 지정합니다. 기본값은 'l2'이며, 'l1', 'l2', 'elasticnet', 'none' 중 하나를 선택할 수 있습니다.
#    - `'l1'`: L1 규제(Lasso)
#    - `'l2'`: L2 규제(Ridge)
#    - `'elasticnet'`: L1과 L2의 결합
#    - `'none'`: 규제 없음
# 
# 2. **dual**: 듀얼 형식을 사용할지 여부를 지정합니다. 기본값은 `False`입니다. 주로 샘플 수가 특성 수보다 많은 경우에 `False`로 설정합니다.
# 
# 3. **tol**: 종료 기준을 설정하는 데 사용되는 허용 오차입니다. 기본값은 `1e-4`입니다.
# 
# 4. **C**: 규제 강도를 설정합니다. 값이 작을수록 강한 규제를 의미합니다. 기본값은 `1.0`입니다.
# 
# 5. **fit_intercept**: 절편을 추가할지 여부를 지정합니다. 기본값은 `True`입니다.
# 
# 6. **intercept_scaling**: `fit_intercept=True`일 때 절편 항에 대한 스케일링 값입니다. 기본값은 `1`입니다.
# 
# 7. **class_weight**: 클래스 가중치를 지정합니다. 기본값은 `None`이며, `balanced`로 설정할 수 있습니다.
# 
# 8. **random_state**: 난수 생성기를 설정합니다. 결과의 재현성을 위해 사용됩니다.
# 
# 9. **solver**: 최적화 알고리즘을 지정합니다. 기본값은 `'lbfgs'`입니다. 가능한 값으로는 `'newton-cg'`, `'lbfgs'`, `'liblinear'`, `'sag'`, `'saga'`가 있습니다.
#    - `'newton-cg'`: Newton의 방법 변형
#    - `'lbfgs'`: Broyden-Fletcher-Goldfarb-Shanno (BFGS) 알고리즘 변형
#    - `'liblinear'`: 작은 데이터셋에 적합
#    - `'sag'`: 대규모 데이터셋에 적합
#    - `'saga'`: 매우 대규모 데이터셋에 적합
# 
# 10. **max_iter**: 최대 반복 횟수입니다. 기본값은 `100`입니다.
# 
# 11. **multi_class**: 다중 클래스 설정 방법을 지정합니다. 기본값은 `'auto'`입니다.
#     - `'auto'`: 이진 분류에서는 이진 로지스틱 회귀를 사용하고, 다중 분류에서는 OvR을 사용합니다.
#     - `'ovr'`: 일대다(One-vs-Rest) 전략
#     - `'multinomial'`: 다항 로지스틱 회귀
# 
# 12. **verbose**: 출력할 로깅 정보의 양을 설정합니다. 기본값은 `0`입니다.
# 
# 13. **warm_start**: 이전 학습의 결과를 초기화에 사용할지 여부를 지정합니다. 기본값은 `False`입니다.
# 
# 14. **n_jobs**: 병렬 작업에 사용할 CPU 코어 수를 지정합니다. 기본값은 `None`입니다.

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report


# - 타이타닉 데이터셋으로 1차로 성능 평가
# - 위에 있는 하이퍼파라미터를 조금 조정해서 성능을 올려보는 것

# In[13]:


#타이타닉 불러오기
df=sns.load_dataset('titanic')

# In[14]:


#결측치 정리

df=df.dropna(subset=['age','embarked'])

#데이터 전처리
df['sex'] = df['sex'].map({'male':0, 'female':1})
df =pd.get_dummies(df, columns = ['embarked'],drop_first=True)


# In[15]:


# 특성 x, y 분리
X= df[['sex','age','fare','embarked_Q','embarked_S']]
y = df['survived']

# In[18]:


# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=111)


# In[20]:


#특성 스케일링
scaler=StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


# In[22]:


#로지스틱 회귀 모델 실습
#Base Model
model=LogisticRegression()
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)
y_pred_prob=model.predict_proba(X_test)[:,1]

# In[25]:


##성능평가
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc =roc_auc_score(y_test, y_pred)

# In[28]:


## 혼동행렬, 분류 보고서 출력
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# In[31]:


print(conf_matrix)
print(class_report)
print(accuracy_score(y_test, y_pred))

# ### 하이퍼파라미터 튜닝을 통해 성능 개선

# In[32]:


#로지스틱 회귀 모델 실습
#Base Model
model_t1=LogisticRegression(
            penalty='l2',
            solver='liblinear',
            class_weight = 'balanced',
            max_iter = 100)
model_t1.fit(X_train, y_train)

# 예측
y_pred = model_t1.predict(X_test)
y_pred_prob=model_t1.predict_proba(X_test)[:,1]

# In[34]:


##하이퍼파라미터 튜닝 한 성능평가
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc =roc_auc_score(y_test, y_pred)

# In[35]:


## 혼동행렬, 분류 보고서 출력
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# In[36]:


print(conf_matrix)
print(class_report)
print(accuracy_score(y_test, y_pred))

# In[37]:


df

# ## BMI 예측 진행

# In[46]:


#BMI 데이터로 분석!
df=pd.read_csv('heart_2020_cleaned.csv')

# In[47]:


# 사용할 컬럼 정리
df2 =pd.get_dummies(df, columns = ['HeartDisease','Smoking','AlcoholDrinking','Stroke','DiffWalking','Sex','AgeCategory','Race','Diabetic','PhysicalActivity','GenHealth','Asthma','KidneyDisease','SkinCancer'], drop_first=True)

# In[49]:


df2.columns

# In[51]:


##스케일링 작업
df_num = df2[['BMI','PhysicalHealth','MentalHealth','SleepTime']]
df_nom = df2.drop(['BMI','PhysicalHealth','MentalHealth','SleepTime'],axis=1)

# In[53]:


## Standard 스케일링
scaler=StandardScaler()
df_scaler=scaler.fit_transform(df_num)

# In[57]:


df_num2 =pd.DataFrame(data= df_scaler, columns= df_num.columns)

# In[61]:


df_new=pd.concat([df_num2, df_nom], axis=1)

# In[62]:


df_new

# In[63]:


X = df_new.drop(['HeartDisease_Yes'], axis=1)
y = df_new[['HeartDisease_Yes']]

# In[66]:


## train,test 분리
X_train, X_test, y_train, y_test =train_test_split(X,y, test_size=0.3, random_state=111)

# In[68]:


print(len(X_train), 'train 수')
print(len(X_test), 'test 수')

# In[69]:


#Base Model
#심장병에대한 예측
model=LogisticRegression()
model.fit(X_train, y_train)

# In[70]:


# 예측
y_pred = model.predict(X_test)
y_pred_prob=model.predict_proba(X_test)[:,1]

##성능평가
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc =roc_auc_score(y_test, y_pred)

## 혼동행렬, 분류 보고서 출력
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(conf_matrix)
print(class_report)
print(accuracy_score(y_test, y_pred))

# In[72]:


# 두 개의 성능 차이를 보고 
print('학습셋 모델 정확도', model.score(X_train, y_train))
print('테스트셋 모델 정확도', model.score(X_test, y_test))

# - recall, f1-score
# - 0.1 0.17

# - accuracy 높은데, recall, precision, f1 은 상대적으로 낮다.

# ### 클래스이 대한 불균형으로 인해서 성능이 정확도는 높게 나오더라도 다른 지표들이 같이 보고 확인해야 한다.

# In[73]:


y.value_counts()

# In[80]:


from imblearn.under_sampling import *

# In[75]:


X_train_re =X_train.copy()
y_train_re =y_train.copy()

# In[81]:


## 임시로 데이터셋 name

X_temp_name = ['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10',
            'X11','X12','X13','X14','X15','X16','X17','X18','X19','X20',
            'X21','X22','X23','X24','X25','X26','X27','X28','X29','X30',
            'X31','X32','X33','X34','X35','X36','X37']
y_temp_name = ['y1']


X_train_re.columns = X_temp_name
y_train_re.columns = y_temp_name

X_train_re.head()


# In[83]:


# 언더샘플링
X_train_under, y_train_under =RandomUnderSampler(random_state=111).fit_resample(X_train_re, y_train_re)

# In[86]:


print(X_train_re.shape, y_train_re.shape)

# In[87]:


print(X_train_under.shape, y_train_under.shape)

# In[91]:


## 언더샘플링 전
y_train_re['y1'].value_counts()

# In[89]:


## 언더샘플링 후
## y값의 분포 확인

y_train_under['y1'].value_counts()

# In[92]:


## 컬럼명 복구 
X_train_under.columns = list(X_train)

# In[94]:


## 컬럼명 복구
y_train_under.columns = list(y_train)

# In[96]:


#Base Model
#심장병에대한 예측
model=LogisticRegression()
model.fit(X_train_under, y_train_under)

# In[97]:


# 예측
y_pred = model.predict(X_test)
y_pred_prob=model.predict_proba(X_test)[:,1]

##성능평가
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc =roc_auc_score(y_test, y_pred)

## 혼동행렬, 분류 보고서 출력
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(conf_matrix)
print(class_report)
print(accuracy_score(y_test, y_pred))

# - [87019   729]
# -  [ 7355   836]]
# >               precision    recall  f1-score   support
# 
#                0       0.92      0.99      0.96     87748
#                1       0.53      0.10      0.17      8191
# 
#         accuracy                           0.92     95939
#        macro avg       0.73      0.55      0.56     95939
#     weighted avg       0.89      0.92      0.89     95939
# 
# - 0.9157381252670967

# In[98]:


# 계수 값도 확인 가능

model.coef_

# In[99]:


## 로지스틱회귀도 stats 모델에서 Logit 사용해서 summary를 확인할 수 있다.
## 공식적으로 정해진 건 없어서 

import statsmodels.api as sm

# In[100]:


ml1=sm.Logit(y_train_under, X_train_under)

# In[102]:


res=ml1.fit(method='newton') # 최적화 방법론 

# In[103]:


res.summary()

# ### 모델 요약 통계 항목 설명
# 
# 1. **Dep. Variable**: 종속 변수(타겟 변수)의 이름입니다. 여기서는 `HeartDisease_Yes`로, 심장 질환 여부를 나타냅니다.
# 
# 2. **No. Observations**: 모델에 사용된 관측치(데이터 포인트)의 총 개수입니다. 여기서는 223,856개입니다.
# 
# 3. **Model**: 사용된 모델의 유형입니다. 여기서는 로지스틱 회귀(`Logit`)입니다.
# 
# 4. **Df Residuals**: 잔차의 자유도입니다. 이는 총 관측치 수에서 모델에 사용된 파라미터 수를 뺀 값입니다. 여기서는 223,819입니다.
# 
# 5. **Method**: 모델 추정 방법입니다. 여기서는 최대 우도 추정(MLE, Maximum Likelihood Estimation) 방법을 사용했습니다.
# 
# 6. **Df Model**: 모델의 자유도입니다. 이는 사용된 독립 변수의 수입니다. 여기서는 36입니다.
# 
# 7. **Date**: 모델이 적합된 날짜입니다. 여기서는 2024년 1월 21일입니다.
# 
# 8. **Pseudo R-squ.**: 의사 R-제곱(Pseudo R-squared) 값입니다. 이는 모델의 설명력을 나타내는 지표입니다. 여기서는 0.1811입니다.
# 
# 9. **Time**: 모델이 적합된 시간입니다. 여기서는 20:50:12입니다.
# 
# 10. **Log-Likelihood**: 로그 우도(Log-Likelihood) 값입니다. 이는 모델이 데이터를 얼마나 잘 설명하는지 나타냅니다. 값이 클수록 모델이 데이터를 잘 설명합니다. 여기서는 -53,613입니다.
# 
# 11. **converged**: 모델이 수렴했는지 여부입니다. 여기서는 `True`로, 모델이 성공적으로 수렴했음을 나타냅니다.
# 
# 12. **LL-Null**: Null 모델의 로그 우도 값입니다. 여기서는 -65,466입니다.
# 
# 13. **Covariance Type**: 공분산 행렬의 유형입니다. 여기서는 `nonrobust`입니다.
# 
# 14. **LLR p-value**: 우도비 검정의 p-값입니다. 이는 모델의 유의성을 테스트합니다. p-값이 매우 낮을 경우(일반적으로 0.05보다 작으면), 모델이 통계적으로 유의함을 나타냅니다. 여기서는 p-값이 0.000으로, 모델이 통계적으로 유의합니다.
# 
# ### 회귀 계수 해석
# 
# 회귀 계수 표는 각 독립 변수에 대한 정보를 제공합니다. 여기서는 `BMI` 변수를 예로 들어 해석해보겠습니다.
# 
# - **coef**: 회귀 계수 값입니다. `BMI` 변수의 회귀 계수는 0.0855입니다. 이는 `BMI`가 1 단위 증가할 때 심장 질환 발생 확률의 로그 오즈가 0.0855만큼 증가함을 의미합니다.
# - **std err**: 회귀 계수의 표준 오차입니다. `BMI` 변수의 표준 오차는 0.010입니다.
# - **z**: z-값입니다. 이는 회귀 계수를 표준 오차로 나눈 값입니다. `BMI` 변수의 z-값은 8.739입니다.
# - **P>|z|**: p-값입니다. 이는 해당 회귀 계수가 0이라는 귀무 가설을 검정합니다. `BMI` 변수의 p-값은 0.000으로, 매우 유의미하다는 것을 나타냅니다.
# - **[0.025 0.975]**: 회귀 계수의 95% 신뢰 구간입니다. `BMI` 변수의 신뢰 구간은 [0.066, 0.105]입니다.
# 
# ### 요약
# 
# - `BMI` 변수는 심장 질환 발생 확률과 유의미한 양의 상관관계를 가집니다.
# - 모델이 통계적으로 유의하며, `BMI`를 포함한 여러 변수들이 심장 질환 발생에 영향을 미치는 것으로 보입니다.
# 
# ### 전체 해석
# 
# - **모델의 설명력**: Pseudo R-squared 값이 0.1811로, 모델이 데이터의 약 18.11%를 설명합니다. 이는 로지스틱 회귀에서 흔히 낮게 나오는 값이며, 모델이 충분히 유의미한지 다른 지표와 함께 검토해야 합니다.
# - **변수의 유의성**: `BMI` 변수는 p-값이 0.000으로 매우 유의미하며, 심장 질환 발생에 긍정적인 영향을 미친다고 해석할 수 있습니다.

# In[ ]:



