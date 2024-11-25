#!/usr/bin/env python
# coding: utf-8

# ## Randomforest 하이퍼파라미터 정리
# 1. **n_estimators**: 
#    - 결정 트리의 개수를 의미합니다.
#    - 기본값은 100입니다.
#    - 트리의 수를 늘릴수록 모델의 성능은 일반적으로 향상되지만, 계산 비용도 증가합니다.
# 
# 2. **criterion**: 
#    - 분할의 품질을 측정하는 기능입니다.
#    - 분류에 사용되는 기준은 "gini" (지니 불순도) 또는 "entropy" (정보 이득)이 될 수 있습니다.
#    - 기본값은 "gini"입니다.
# 
# 3. **max_depth**: 
#    - 트리의 최대 깊이입니다.
#    - 너무 깊은 트리는 과적합을 일으킬 수 있습니다.
#    - 기본값은 None으로, 모든 리프 노드가 순수하거나 min_samples_split보다 적은 수의 샘플을 가질 때까지 노드가 확장됩니다.
# 
# 4. **min_samples_split**: 
#    - 노드를 분할하기 위해 필요한 최소 샘플 수입니다.
#    - 기본값은 2입니다.
# 
# 5. **min_samples_leaf**: 
#    - 리프 노드가 가져야 하는 최소 샘플 수입니다.
#    - 기본값은 1입니다.
# 
# 6. **min_weight_fraction_leaf**: 
#    - 리프 노드가 가지고 있어야 하는 샘플의 최소 가중치 합의 비율입니다.
#    - 기본값은 0입니다.
# 
# 7. **max_features**: 
#    - 각 분할에서 고려할 특성의 수입니다.
#    - 옵션은 "auto", "sqrt", "log2" 또는 None 또는 정수입니다.
#    - 기본값은 "auto"로, 분류에서는 sqrt(n_features), 회귀에서는 n_features입니다.
# 
# 8. **max_leaf_nodes**: 
#    - 리프 노드의 최대 수입니다.
#    - 기본값은 None으로, 리프 노드의 수에 제한이 없습니다.
# 
# 9. **min_impurity_decrease**: 
#    - 분할로 인한 불순도 감소량의 최소값입니다.
#    - 기본값은 0입니다.
# 
# 10. **bootstrap**: 
#     - 부트스트랩 샘플을 사용할지 여부입니다.
#     - 기본값은 True입니다.
# 
# 11. **oob_score**: 
#     - 일반적으로 사용하지 않는 샘플을 사용하여 오류를 추정할지 여부입니다.
#     - 기본값은 False입니다.
# 
# 12. **n_jobs**: 
#     - 핏(fit)과 예측(predict)을 위해 병렬로 실행할 작업 수입니다.
#     - 기본값은 None으로, 1개의 작업만 사용합니다. -1로 설정하면 모든 프로세서를 사용합니다.
# 
# 13. **random_state**: 
#     - 난수 seed 설정입니다.
#     - 여러 번 실행해도 동일한 결과를 얻기 위해 사용됩니다.
# 
# 14. **verbose**: 
#     - 실행 과정 중 메시지를 출력할지 여부를 설정합니다.
#     - 기본값은 0으로, 메시지를 출력하지 않습니다.
# 
# 15. **warm_start**: 
#     - 이전 호출의 솔루션을 재사용하여 더 많은 추정기를 학습시킬지 여부입니다.
#     - 기본값은 False입니다.
# 
# 16. **class_weight**: 
#     - 클래스 가중치입니다.
#     - 기본값은 None으로,

# ### 1. **`VotingClassifier` 하이퍼파라미터 (분류)**
# 
# ```python
# from sklearn.ensemble import VotingClassifier
# ```
# 
# - **`estimators`**: 
#   - 설명: 앙상블에 포함될 예측기(모델) 목록을 지정합니다. 이 목록은 (name, estimator) 쌍으로 구성된 리스트여야 합니다.
#   - 예: `[('decision_tree', dt), ('random_forest', rf)]`
# 
# - **`voting`**:
#   - 설명: 보팅 유형을 지정합니다.
#     - `'hard'`: 다수결 투표를 사용하여 예측 결과를 결정합니다.
#     - `'soft'`: 예측 확률의 평균을 사용하여 예측 결과를 결정합니다.
#   - 기본값: `'hard'`
#   - 예: `'soft'`, `'hard'`
# 
# - **`weights`**:
#   - 설명: 각 예측기에 대한 가중치를 지정합니다. 가중치는 `voting='soft'`일 때만 적용됩니다. 가중치가 높을수록 해당 모델의 예측이 더 중요한 역할을 합니다.
#   - 기본값: `None` (모든 모델에 동일한 가중치가 적용됨)
#   - 예: `[1, 2]` (첫 번째 모델에 1, 두 번째 모델에 2의 가중치를 부여)
# 
# - **`n_jobs`**:
#   - 설명: 앙상블을 병렬로 학습할 때 사용할 CPU 코어 수를 지정합니다.
#   - 기본값: `None` (1개의 코어 사용)
#   - 예: `-1` (사용 가능한 모든 코어를 사용)
# 
# - **`flatten_transform`**:
#   - 설명: 각 예측기의 `transform` 결과를 2D 배열로 변환할지 여부를 지정합니다. `voting='soft'`일 때만 적용됩니다.
#   - 기본값: `True`
#   - 예: `True`, `False`
# 
# - **`verbose`**:
#   - 설명: 앙상블 학습 과정의 진행 상황을 출력할지 여부를 지정합니다.
#   - 기본값: `False`
#   - 예: `True`, `False`
# 
# ### 2. **`VotingRegressor` 하이퍼파라미터 (회귀)**
# 
# ```python
# from sklearn.ensemble import VotingRegressor
# ```
# 
# - **`estimators`**: 
#   - 설명: 앙상블에 포함될 예측기(모델) 목록을 지정합니다. 이 목록은 (name, estimator) 쌍으로 구성된 리스트여야 합니다.
#   - 예: `[('linear', lr), ('rf', rf)]`
# 
# - **`weights`**:
#   - 설명: 각 예측기에 대한 가중치를 지정합니다. 예측기의 예측 결과에 대한 가중 평균을 계산합니다.
#   - 기본값: `None` (모든 모델에 동일한 가중치가 적용됨)
#   - 예: `[1, 2]` (첫 번째 모델에 1, 두 번째 모델에 2의 가중치를 부여)
# 
# - **`n_jobs`**:
#   - 설명: 앙상블을 병렬로 학습할 때 사용할 CPU 코어 수를 지정합니다.
#   - 기본값: `None` (1개의 코어 사용)
#   - 예: `-1` (사용 가능한 모든 코어를 사용)
# 
# - **`verbose`**:
#   - 설명: 앙상블 학습 과정의 진행 상황을 출력할지 여부를 지정합니다.
#   - 기본값: `False`
#   - 예: `True`, `False`
# 

# In[8]:


import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFECV, VarianceThreshold
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 타이타닉 데이터셋 로드
titanic = sns.load_dataset('titanic')

# 결측값 처리 (예: Age, Embarked의 결측값을 평균 또는 최빈값으로 대체)
if 'embarked' in titanic.columns:
    titanic['age'].fillna(titanic['age'].mean(), inplace=True)
    titanic['embarked'].fillna(titanic['embarked'].mode()[0], inplace=True)
    titanic['embark_town'].fillna(titanic['embark_town'].mode()[0], inplace=True)
    titanic['fare'].fillna(titanic['fare'].mean(), inplace=True)
else:
    print("The 'embarked' column is not found in the dataset.")

# 범주형 변수 인코딩
titanic = pd.get_dummies(titanic, columns=['sex', 'embarked', 'class', 'who', 'embark_town', 'deck', 'alone'], drop_first=True)

# 불필요한 열 제거
titanic.drop(['alive', 'adult_male', 'parch', 'sibsp'], axis=1, inplace=True)

# 입력 변수(X)와 타겟 변수(y) 정의
X = titanic.drop('survived', axis=1)
y = titanic['survived']

# - base Rf 사용 코드

# In[11]:


##train,test 나누기
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,  random_state=111)


# 랜덤포레스트 가지고 오기
rf_model = RandomForestClassifier(n_estimators=100,random_state=111)

# 모델학습
rf_model.fit(X_train,y_train)

# In[17]:


## 예측
y_pred = rf_model.predict(X_test)

## 평가
accuracy =accuracy_score(y_test, y_pred)
print(f'acc:{accuracy:.4f}')

print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))

# In[23]:


## 피처의 중요도 출력
f_im=rf_model.feature_importances_

for i, feature in enumerate(X.columns):
    print(f'{feature}:{f_im[i]:.4f}')

# ## OOB score 도 확인하고, 좀 더 search를 통해 최적의 하이퍼 파라미터를 확인
# - 교차검증
# - search - BayesianOptimization
# - 기본적인 하이퍼파라미터 조정
# - OOB

# In[30]:


## 성능평가 함수 (추가한 내용들)
def rf_eval(n_estimators, max_depth, min_samples_split, min_samples_leaf):
    #랜덤포레스트르 정의
    clf = RandomForestClassifier(
        n_estimators = int(n_estimators),
        max_depth = int(max_depth),
        min_samples_split = int(min_samples_split),
        min_samples_leaf = int(min_samples_leaf),
        oob_score = True, ##OOB 계산
        random_state=111    
    )
    
    return cross_val_score(clf, X_train, y_train, scoring='accuracy',cv=3).mean()



## 베이지안 최적화 설정
optimizer =BayesianOptimization(
    f = rf_eval,
    pbounds={
        'n_estimators':(100,400),
        'max_depth':(10,20),
        'min_samples_split':(2,10),
        'min_samples_leaf':(1,20)
    },
    random_state =111
    )

## 베이지안 최적화 수행

optimizer.maximize(
    n_iter = 20, # 최적화 반복 횟수
    init_points = 5 # 초기 랜덤 탐색 횟수
    )

## 최적의 하이퍼파라미터 성능 출력
print('Best parameters: ', optimizer.max['params'])
print('Best acuuracy: ', optimizer.max['target'])

## 최적의 하이퍼파라미터로 랜덤포레스트를 다시 학습 (OOB 점수까지 같이 계산)
best_params=optimizer.max['params']
rf_best =RandomForestClassifier(
        n_estimators = int(best_params['n_estimators']),
        max_depth = int(best_params['max_depth']),
        min_samples_split = int(best_params['min_samples_split']),
        min_samples_leaf = int(best_params['min_samples_leaf']),
        oob_score = True, ##OOB 계산
        random_state=111
        ) 

# 모델학습
rf_best.fit(X_train,y_train)

## 예측
y_pred = rf_best.predict(X_test)

## 평가
accuracy =accuracy_score(y_test, y_pred)
print(f'acc:{accuracy:.4f}')

print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))

# max_depth': 10.355602803953097, 'min_samples_leaf': 8.13043203124824, 'min_samples_split': 9.690576415089538, 'n_estimators': 184.05919539312265

# Best acuuracy:  0.8216501790589654
# 

# In[40]:


## 좀 더 시각적으로 잘 확인
feature_importances =rf_best.feature_importances_
sort_indices=np.argsort(feature_importances)[::-1]
sort_feature = X.columns[sort_indices]
sorted_imporatances =feature_importances[sort_indices]

# In[43]:


plt.barh(sort_feature, sorted_imporatances, color='skyblue')

# In[45]:


## learning_curve
## train,test 학습에대한 곡선 그리는 것

from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(
    RandomForestClassifier(n_estimators=100,random_state=111),
    X_train, y_train, cv=3, n_jobs=-1, train_sizes = np.linspace(0.1, 1.0,5)
    )



# In[48]:


#학습곡선 그리기

plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training Score')
plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Test(cross_val) Score')
plt.legend()
plt.show()

# In[49]:


from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(
   RandomForestClassifier(
        n_estimators = int(best_params['n_estimators']),
        max_depth = int(best_params['max_depth']),
        min_samples_split = int(best_params['min_samples_split']),
        min_samples_leaf = int(best_params['min_samples_leaf']),
        oob_score = True, ##OOB 계산
        random_state=111),
    X_train, y_train, cv=3, n_jobs=-1, train_sizes = np.linspace(0.1, 1.0,5)
    )



# In[51]:


#학습곡선 그리기

plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training Score')
plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Test(cross_val) Score')
plt.legend()
plt.show()

# ## Decision_Tree 비교 진행
# - pipeline 모델성능을 비교
# - (코드로 공유하는 쪽으로)

# ## Voting
# - from sklearn.ensemble import VotingClassifier

# In[53]:


## 데이터 분할은 위에 코드를 그대로 사용한다
X_train

# - 두 개의 모델을 사용해서 voting
#     - RandomForestClassifier
#     - DecisionTreeClassifier
#     - voting에 사용할 2개 모델
#     - 여러분은 나중에 원하시면 여러 개 사용하면 된다.
#     - 방법만 알면 사용하는 건 하나도 어렵지 않다.

# In[56]:


## 모델 2개를 불러오자
dt = DecisionTreeClassifier(random_state=111)
rf = RandomForestClassifier(n_estimators=100,random_state=111)


# In[57]:


dt

# In[58]:


rf

# In[61]:


#hard voting
hard_voting_clf =VotingClassifier(
    estimators = [('Decision_tree', dt), ('Random_Forest',rf)],
    voting='hard'
    )

#soft voting
soft_voting_clf =VotingClassifier(
    estimators = [('Decision_tree', dt), ('Random_Forest',rf)],
    voting='soft'
    )

# In[60]:


hard_voting_clf

# In[62]:


soft_voting_clf

# In[63]:


## voting 학습
## voting 예측
hard_voting_clf.fit(X_train,y_train)
y_pred_hard=hard_voting_clf.predict(X_test)

soft_voting_clf.fit(X_train,y_train)
y_pred_soft=soft_voting_clf.predict(X_test)

# In[66]:


## Hard voting 평가

accuracy =accuracy_score(y_test, y_pred_hard)
print(f'acc:{accuracy:.4f}')

print(classification_report(y_test, y_pred_hard))

print(confusion_matrix(y_test, y_pred_hard))

# In[68]:


## soft voting 평가

accuracy =accuracy_score(y_test, y_pred_soft)
print(f'acc:{accuracy:.4f}')

print(classification_report(y_test, y_pred_soft))

print(confusion_matrix(y_test, y_pred_soft))

# In[69]:


## 모델 2개를 불러오자
dt_best = DecisionTreeClassifier(max_depth=10, random_state=111)
rf_best = RandomForestClassifier(
        n_estimators = int(best_params['n_estimators']),
        max_depth = int(best_params['max_depth']),
        min_samples_split = int(best_params['min_samples_split']),
        min_samples_leaf = int(best_params['min_samples_leaf']),
        oob_score = True, ##OOB 계산
        random_state=111)


# In[73]:


#hard voting
hard_voting_clf_best =VotingClassifier(
    estimators = [('Decision_tree', dt_best), ('Random_Forest',rf_best)],
    voting='hard'
    )

#soft voting
soft_voting_clf_best =VotingClassifier(
    estimators = [('Decision_tree', dt_best), ('Random_Forest',rf_best)],
    voting='soft'
    )

# In[74]:


## voting 학습
## voting 예측
hard_voting_clf_best.fit(X_train,y_train)
y_pred_hard=hard_voting_clf_best.predict(X_test)

soft_voting_clf_best.fit(X_train,y_train)
y_pred_soft=soft_voting_clf_best.predict(X_test)

# In[77]:


## Hard voting 평가

accuracy =accuracy_score(y_test, y_pred_hard)
print(f'acc:{accuracy:.4f}')

print(classification_report(y_test, y_pred_hard))

print(confusion_matrix(y_test, y_pred_hard))

# In[78]:


## soft voting 평가

accuracy =accuracy_score(y_test, y_pred_soft)
print(f'acc:{accuracy:.4f}')

print(classification_report(y_test, y_pred_soft))

print(confusion_matrix(y_test, y_pred_soft))

# In[ ]:



