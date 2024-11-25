#!/usr/bin/env python
# coding: utf-8

# ### 다항회귀
# - 회귀가 독립변수의 단항식이 아닌 2차, 3차 방정식과 같은 다항식으로 표형된 것 
# - 다항(Polynomial) 회귀 
# - 다항회귀는 비선형회귀가 아닌, 선형회귀
# - 회귀에서 선형 회귀 / 비선형 회귀를 나누는 기준은 회귀 계수가 선형/비선형인지에 따라 다른 것이지 독립변수의 선형/비선형 여부와는 상관 없음
# - 다항 회귀 (Polynomial Regression): 종속 변수와 독립 변수 간의 비선형 관계를 다항식 형태로 모델링하는 방법
# - 다변량 다항 회귀 (Multivariate Polynomial Regression): 여러 개의 독립 변수와 그들의 다항 항을 포함하여 종속 변수와의 관계를 모델링하는 방법
# - 다항회귀는 과적합이 될 수 있으므로 규제가 필요하다.

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# 데이터 생성
np.random.seed(0)
x = 2 - 3 * np.random.normal(0, 1, 100)
y = x - 2 * (x ** 2) + np.random.normal(-3, 3, 100)

# 다항 회귀 모델 생성
polynomial_features = PolynomialFeatures(degree=2)
x_poly = polynomial_features.fit_transform(x.reshape(-1, 1))

model = LinearRegression()
model.fit(x_poly, y)
y_poly_pred = model.predict(x_poly)

# 시각화
plt.scatter(x, y, s=10)
plt.plot(np.sort(x), y_poly_pred[np.argsort(x)], color='m')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Polynomial Regression')
plt.show()


# ### 다중회귀
# - 다중회귀는 하나의 종속 변수와 둘 이상의 독립 변수 간의 관계를 분석하는 회귀분석
# - 독립변수들간의 다중공선성 (Multicollinearity)는 불안정성을 초래할 수 있음
# - VIF(Variance Inflation Factor)를 사용하여 다중공선성을 진단

# In[2]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 데이터 생성
np.random.seed(0)
x1 = np.random.rand(100)
x2 = np.random.rand(100)
y = 3 + 2 * x1 + 4 * x2 + 1.5 * x1**2 + 0.5 * x2**2 + 2 * x1 * x2 + np.random.normal(0, 0.5, 100)

# 데이터프레임 생성
df = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})

# 독립 변수와 종속 변수 분리
X = df[['x1', 'x2']]
y = df['y']

# 다항 특성 생성
polynomial_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = polynomial_features.fit_transform(X)

# 다항 회귀 모델 생성 및 학습
model = LinearRegression()
model.fit(X_poly, y)

# 회귀 계수 출력
print('Intercept:', model.intercept_)
print('Coefficients:', model.coef_)

# 예측
y_poly_pred = model.predict(X_poly)

# 성능 평가
from sklearn.metrics import mean_squared_error, r2_score
print('Mean Squared Error:', mean_squared_error(y, y_poly_pred))
print('R^2 Score:', r2_score(y, y_poly_pred))

# 시각화 (2차원 공간에서의 예측 시각화)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 실제 데이터
scat = ax.scatter(df['x1'], df['x2'], df['y'], color='b', label='Actual Data')

# 예측값 시각화
x1_grid, x2_grid = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
X_grid_poly = polynomial_features.transform(np.column_stack([x1_grid.ravel(), x2_grid.ravel()]))
y_grid_pred = model.predict(X_grid_poly).reshape(x1_grid.shape)

# 예측 표면
surf = ax.plot_surface(x1_grid, x2_grid, y_grid_pred, color='m', alpha=0.5)

# 축 레이블
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
plt.title('Multivariate Polynomial Regression')

# 범례 추가
actual_proxy = plt.Line2D([0], [0], linestyle="none", marker='o', color='b')
predicted_proxy = plt.Line2D([0], [0], linestyle="none", marker='o', color='m', alpha=0.5)
ax.legend([actual_proxy, predicted_proxy], ['Actual Data', 'Predicted Surface'], numpoints=1)

plt.show()

# In[3]:


import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import matplotlib.pyplot as plt

# 캘리포니아 주택 가격 데이터 로드
california = fetch_california_housing()

# 데이터프레임 생성
df = pd.DataFrame(california.data, columns=california.feature_names)
df['MedHouseVal'] = california.target


# ### 단순한 선형회귀를 통해 
# - MSE, MAE, R2, OLS 비교하기
# 
# ### 다항회귀 2차식으로 진행하게 되는 경우
# - MSE, MAE, R2, OLS 같이 비교하기!
#  

# In[11]:


df

# In[46]:


#MedHouseVal y값
#MedInc 수입의 중앙값 


## 단순 선형회귀 분석을 진행

X = df[['MedInc']]
y = df['MedHouseVal']

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=111)

# In[28]:


## 단순선형회귀
linear_model =LinearRegression()
linear_model.fit(X_trian, y_train)
y_pred_linear = linear_model.predict(X_test) #예측값이 출력

## OLS 분석
X_train_ols = sm.add_constant(X_train)
X_test_ols = sm.add_constant(X_test)
ols_model = sm.OLS(y_train,X_train_ols).fit()
y_pred_ols = ols_model.predict(X_test_ols)


##평가지표를 불러오기

mse_linear =mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

mse_ols_linear = mean_squared_error(y_test, y_pred_ols)
r2_ols_linear = r2_score(y_test, y_pred_ols)


# In[32]:


#ols, sklearn 동일한 값이 출력
print(mse_linear)
print(mse_ols_linear)

# In[31]:


print(r2_linear)
print(r2_ols_linear)

# In[53]:


#다항회귀 추가 
poly_features =PolynomialFeatures(degree=2) #다항회귀 차수 지정
X_train_poly =poly_features.fit_transform(X_train) # 2차 다항식으로 변환하여 학습
X_test_poly = poly_features.transform(np.array(X_test).reshape(-1,1))# 자료타입으로 인한 에러가 발생할 수 있음, 버전으로도 발생할 수 있음

#2차식으로 바꾼 데이터를 학습
poly_model =LinearRegression()
poly_model.fit(X_train_poly, y_train)
y_pred_poly = poly_model.predict(X_test_poly) #예측값이 출력 2차항으로


# In[56]:


##2차 다항식 평가지표
##평가지표를 불러오기

mse_poly =mean_squared_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)

# In[59]:


## 단순선형회귀
print(mse_linear)
print(r2_linear)
print('--------------')
# 2차 다항식으로 
print(mse_poly)
print(r2_poly)

# - 성능에서는 다항식이 조금 더 좋은 성능을 보였다.
# - 과적합이 될 확률이 높기 때문에 규제에 대한 부분도 고려해야 한다.

# In[62]:


poly_ols_model=sm.OLS(y_train, X_train_poly).fit()

# In[60]:


ols_model.summary()

# In[64]:


poly_ols_model.summary()

# ### 피처를 2개 이상 사용하게 되는 경우
# - 그랬을 때 성능에 대해서 확인

# In[70]:


#2개의 피처를 선택
#2개의 피처 기준은 피처셀렉션을 통해 선정하여 2개를 추가해서 다중회귀로 분석
X=df[['MedInc','AveRooms']]
y

# In[71]:


# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=111)

# In[74]:


X_test

# In[75]:


## 다중회귀학습
multiple_model =LinearRegression()
multiple_model.fit(X_train, y_train) # 독립변수가 2개 이상인 값이 들어가는 것
y_pred_multiple = multiple_model.predict(X_test) #예측값이 출력

# In[78]:


## 다중회귀로 진행했을 때 , 2개의 피처 사용

mse_multiple =mean_squared_error(y_test, y_pred_multiple)
r2_multiple = r2_score(y_test, y_pred_multiple)

# In[79]:


print(mse_multiple)
print(r2_multiple)

# In[80]:


## 단순선형회귀
print(mse_linear)
print(r2_linear)
print('--------------')
# 2차 다항식으로 
print(mse_poly)
print(r2_poly)

# - 피처를 추가하면서 성능과 R2 의 값이 더 올라갔다. (좋아짐)
# - 어떤 피처를 추가하는가에 따라 이 성능은 떨어질 수도 있다.
# - 이 피처가 스케일링 자체는 하지 않았다.
# - 피처 자체가 연속적인 것인지 잘 확인해야 한다. 
# - 피처가 많고 다항회귀 하면 더 좋은 성능을 만든다 라고 1차원적으로 생각할 순 없다.
# - 해당 모델을 가지고 예측을 해야 한다. 

# In[81]:


multiple_ols_model=sm.OLS(y_train, X_train).fit()

# In[82]:


multiple_ols_model.summary()

# In[ ]:



