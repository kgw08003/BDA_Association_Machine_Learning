#!/usr/bin/env python
# coding: utf-8

# In[18]:


from sklearn.cluster import KMeans

# In[22]:


kmeans_model1 =KMeans(n_clusters = 4, random_state=111) #고객군의 군집을 3개정도만 나눠보기

# In[23]:


kmeans_model1

# In[27]:


# fit_predict 메서드를 이용해서 라벨을 확인할 수 있다.
kmeans_model1.fit_predict(df_mns_sp)

# In[28]:


df['cluster_kn']=  kmeans_model1.fit_predict(df_mns_sp)

# In[29]:


df

# In[38]:


## 클러스터링 된 값들을 groupby 로 볼 수 있다.

cluster_stats= df.groupby('cluster_kn').agg({
    'Gender': 'count',
    'Age':['mean','std'],
    'Annual Income (k$)' :['mean','std'],
    'Spending Score (1-100)':['mean','std']
})

#간단하게 시각화 진행해 보기!

cluster_stats.columns = ['Count','Age_Mean','Age_Std','Income_mean','Income_std','Score_mean','Score_std']

# 시각화
fig, axes = plt.subplots(3,2, figsize=(15,15))

# Age 평균, 표준편차 시각화
sns.barplot(x=cluster_stats.index, y ='Age_Mean', data=cluster_stats, ax =axes[0,0])
axes[0,0].set_title('Average Age')

sns.barplot(x=cluster_stats.index, y ='Age_Std', data=cluster_stats, ax =axes[0,1])
axes[0,1].set_title('Average Age')

sns.barplot(x=cluster_stats.index, y ='Income_mean', data=cluster_stats, ax =axes[1,0])
axes[1,0].set_title('Average Income')

sns.barplot(x=cluster_stats.index, y ='Income_std', data=cluster_stats, ax =axes[1,1])
axes[1,1].set_title('Average Income')

sns.barplot(x=cluster_stats.index, y ='Score_mean', data=cluster_stats, ax =axes[2,0])
axes[2,0].set_title('Average Score')

sns.barplot(x=cluster_stats.index, y ='Score_std', data=cluster_stats, ax =axes[2,1])
axes[2,1].set_title('Average Score')

# In[31]:


cluster_stats

# # RFM 분석
# - RFM 고객 분석하기 위한 피처를 R,F,M 피처를 가지고 고객을 이해하고, 고객을 분석하자라는 분석 방법론
# - R : Recency : 고객 중 가장 최근 구매한 상품 구입일과 현재 기준까지의 기간 
# - F : Frequency : 상품 구매 횟수, 주문 횟수 ( 기준에 따라 다르다. )
# - M : Monetary : 고객의 총 주문 금액
# ---
# - 기준을 잘 정해야 한다. 
# - 도메인에 따라 조금씩 바뀔 수 있다.
# - RFM 피처 외에 추가적인 더해서 분석할 수 있다. 
# - 쿠폰에 대한 사용, 고객의 불만율, 고객의 다양한 제품 구매, 고객의 지속기간, 손익 등 피처로 추가해서
# - 4개의 피처로도 분석이 가능하다. 

# In[40]:


retail_df=pd.read_excel(io='online_retail_II.xlsx')

# In[43]:


retail_sp=retail_df[0:10000]

# In[46]:


# 취소반품 고객 제외 
retail_sp=retail_sp[retail_sp['Price']>0]
# 주문 취소했으니 주문 수량도 - 인 경우 제외 
retail_sp=retail_sp[retail_sp['Quantity']>0]


# In[51]:


# 회원인 고객만 분석 가정 
retail_sp=retail_sp[retail_sp['Customer ID'].notnull()]

# In[54]:


# 구매 국가도 영국으로만 지정
retail_sp=retail_sp[retail_sp['Country']=='United Kingdom']

# In[55]:


retail_sp

# - M 피처 
# - Quantity * Price	 = Monetary M피처 만들기

# In[57]:


retail_sp['sales_amount']= retail_sp['Quantity']*retail_sp['Price']

# In[59]:


retail_sp['Customer ID'] = retail_sp['Customer ID'].astype(int)

# - 고객 하나 하나의 지표를 수립하는 것
# - Customer ID별로 groupby를 진행해야 한다. 

# In[61]:


#rfm 지표 만들기


agg_rfm = {
    'InvoiceDate' :'max', #주문일자, 가장 최근이니 max
    'Invoice' : 'count', #카운트 -> 주문 제품 수량 
    'sales_amount' : 'sum' # 전체 주문 금액 sum
}

cust_df =retail_sp.groupby('Customer ID').agg(agg_rfm)

# - Timestamp('2009-12-04 09:31:00') 가장 최근 주문  날짜

# In[62]:


cust_df 

# In[66]:


max(retail_sp.InvoiceDate)

# In[72]:


import datetime as dt 
# R 피처를 추가적으로 만들기 위해서

cust_df['Recency']= cust_df['InvoiceDate'] - dt.datetime(2009,12,31)

# In[77]:


# timedelta 계산된 값을 x.days+1로 나눠서 수치형으로 변환
cust_df['Recency']=cust_df['Recency'].apply(lambda x: x.days+1)

# In[79]:


cust_rfm=cust_df[['Invoice','sales_amount','Recency']]

# In[80]:


cust_rfm

# -----
# - 1.1 기존 데이터의 EDA를 통해 피처들간의 시각화, 통계치 등 분석, 인사이트 정리
#     - 원본 데이터에 대한 분석
#     
# - 1.2 RFM 피처를 만들기 위한 기준을 각자 정해서 기준점 코드로 데이터 전처리
#     - e.g. 반품은 제외하거나 , 영국 고객만 분석하거나, 비회원은 제외하거나 등등
# - 1.3 RFM 피처를 만들어서 실제 통계치로 비교해 보기 
# 
# - 1.4 RFM 피처를 통해 군집화 진행 
#     - 우리가 배운 KMeans를 통해 군집화 진행 
#     - 최적의 군집이 몇 이고? 그런 근거들을 코드와 시각화를 통해 정리
#     
# - 1.5 RFM 피처에 군집된 label 원본데이터 붙이기 
#     - 전체 데이터셋에 label이 형성될 것 
#     
# - 1.6 RFM 피처를 이용하여 붙인 label로 다른 피처들 분석하기 
#     - e.g. 이 고객군들의 주문 제품들은 무엇인지?
#     - 시계열 적으로 볼 때 고객들의 주문 패턴등은 어떤 식으로?
#     - Description에 대한 추가 분석 등을 조금 더 진행하시면 됩니다.
#     
# - 1.7 해당 데이터셋의 고객군들의 RFM 지표의 기초통계치를 정리하고, 시각화를 통해 군집들의 기초통계치 ( 군집들의 RFM 지표를 같이 정리 )
#     - 전체 RFM 지표의 통계치
#     - 우리가 군집한 군집들의 RFM 지표 통계치 
#     - 통계치는 (평균, 중앙값, 표준편차, 최빈값 등등 )

# In[82]:


retail_df

# In[ ]:



