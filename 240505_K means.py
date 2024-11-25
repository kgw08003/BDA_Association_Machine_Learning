#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import KMeans

# 데이터 생성
# 시나리오 1: 잘 분리된 동일 밀도 클러스터
X0, y0 = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)

# 시나리오 2: 다른 밀도의 클러스터
X1, y1 = make_blobs(n_samples=[100, 800], centers=[(-1, 0), (1, 2)], cluster_std=[0.5, 2.5], random_state=42)

# 시나리오 3: 비선형 클러스터 (반달 모양)
X2, y2 = make_moons(n_samples=200, noise=0.05, random_state=42)

# K-means 알고리즘 적용
kmeans0 = KMeans(n_clusters=3, random_state=42).fit(X0)
kmeans1 = KMeans(n_clusters=2, random_state=42).fit(X1)
kmeans2 = KMeans(n_clusters=2, random_state=42).fit(X2)

# 시각화 함수
def plot_clusters(X, y, centroids, title):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='viridis', marker='o', edgecolors='k', alpha=0.5)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='x')
    plt.title(title)
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")

# 그래프 그리기
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plot_clusters(X0, kmeans0.labels_, kmeans0.cluster_centers_, "Scenario 1: Well-separated Clusters")

plt.subplot(1, 3, 2)
plot_clusters(X1, kmeans1.labels_, kmeans1.cluster_centers_, "Scenario 2: Clusters with Different Densities")

plt.subplot(1, 3, 3)
plot_clusters(X2, kmeans2.labels_, kmeans2.cluster_centers_, "Scenario 3: Non-linear Clusters (Moon-shaped)")
plt.show()

# In[ ]:


# 필요한 패키지 설치
#pip install yellowbrick
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.cluster import DBSCAN
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# In[ ]:


#샘플 데이터 출력
df=pd.read_csv('Mall_Customers.csv')

# In[ ]:


df

# ## Kmeans를 이용해서 군집 생성
# - 성별, 나이, 수입, 소비
# - 피처에 대한 이해도 함께 군집분석을 진행하면 좋다.

# In[ ]:


df_sp=df.drop('CustomerID',axis=1)

# In[ ]:


# 군집화를 위한 피처를 정리하기
df_sp

# In[ ]:


sns.set_style('ticks')
sns.pairplot(df_sp, hue='Gender')
plt.show()

# ## 군집화 이전
# ### 스케일링은 필수적으로 해야 한다.
# 
# - 스케일링 방법은 데이터 따라 다르게 사용하시면 됩니다. 이상치를 비교해 보고 다양한 데이터 시각화 작업을 진행 후 확인

# In[ ]:


#인코딩작업
df_km = pd.get_dummies(df_sp, columns = ['Gender'],drop_first=True)

# In[ ]:


#스케일링 불러오기
mns = MinMaxScaler()

df_mns = mns.fit_transform(df_km)

# In[ ]:


# 컬럼 합쳐보기

df_mns_sp = pd.DataFrame(data= df_mns, columns = df_km.columns)

#minmax 스케일링 작업 완료
df_mns_sp

# ## Kmeans 불러오기!
# 
# - init : 초기 중심점을 어떤 식으로 둘 것인지
# - n_cluster : 내가 지정할 클러스터 수
# - n_init = 몇번 반복할 것인가? 중심점 이동하면서
# - random_state : 무작위값 제어하는 시드값
# - max_iter : 알고리즘 수렴 전 최대 몇 번 반복할지

# In[ ]:


kmeans_model1 =KMeans(n_clusters = 3, random_state=111) #고객군의 군집을 3개정도만 나눠보기

# In[ ]:


#kmeans 학습은 fit
#스케일링 한 값을 넣는다.
kmeans_model1.fit(df_mns_sp)


# In[ ]:


#kmeans 제공하는 함수들
#inertia_ SSE 값 출력
#cluster_centers_ 중심 좌표
# n_iter_ 반복횟수 출력
print(kmeans_model1.inertia_)
print(kmeans_model1.cluster_centers_)
print(kmeans_model1.n_iter_)
print(kmeans_model1.labels_) #내가 kmeans 군집의 레이블을 확인할 수 있다. 인덱스기준으로 나열된 내용들

# ## 엘보우 차트를 통해서 군집에 대한 평가를 진행

# In[ ]:


Elbow_ch =KElbowVisualizer(kmeans_model1)
Elbow_ch.fit(df_mns_sp) #데이터 값 그대로 넣기
Elbow_ch.draw() #엘보우 그래프 그리기

# ## 실루엣계수를 통해서 간단하게 시각화

# In[ ]:


# 반복문을 통해 실루엣 계수가 어떤 식으로 변화하는지 체크하기
KMeans_model1={'random_state':111}

# 실루엣 계수를 추가하기
# 군집의 수가 변화하면서 어떤 식으로 값이 변화하는지를 같이 살펴보는 것

sil_coef = []

# 실루엣계수의 그래프 생성

for i in range(2,11):
    kmeans_sil = KMeans(n_clusters = i, **KMeans_model1)
    kmeans_sil.fit(df_mns_sp) #데이터 학습
    score = silhouette_score(df_mns_sp, kmeans_sil.labels_)
    sil_coef.append(score)

plt.plot(range(2,11), sil_coef)
plt.xticks(range(2,11))
plt.show()

# In[ ]:


## 군집별로 같이 시각화를 통해 살펴보자!

fig, ax = plt.subplots(3,2 , figsize=(15,10))

for i in [2,3,4,5,6,7]:
    kmeans_model3 = KMeans(
        n_clusters=i,
        random_state=111 )
    q, mod = divmod(i,2)

# 실루엣계수 시각화를 군집별로 진행

    visual = SilhouetteVisualizer(kmeans_model3,
                                 color = 'yellowbricks',
                                 ax=ax[q-1][mod])

    visual.fit(df_mns_sp) #데이터셋 학습

# In[ ]:



