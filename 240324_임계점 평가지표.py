#!/usr/bin/env python
# coding: utf-8

# ## 클래스를 예측할 때 임계값을 0.5로 기준을 디폴트 값인데,
# - 만약 이 부분을 점점 낮추거나 높이면 어떤 식으로 변화하는지
# - 정밀도, 재현율이 어떤 식으로 변경이 되고 왜 우리는 이 두 지표를 같이 봐야하는지

# In[5]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# In[2]:


df =sns.load_dataset('titanic')

# In[3]:


df_sp=df[['survived','pclass','age','sibsp','parch','fare']]
df_sp.dropna(inplace=True)

# In[6]:


df_sp

# In[7]:


X_train, X_test, y_train, y_test = train_test_split(df_sp.drop('survived',axis=1), df_sp['survived'], test_size=0.3, random_state=111)

# In[8]:


#평가지표 함수
def get_clf_eval(y_test, pred):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    print('오차 행렬')
    print(confusion)
    print('정확도: {0:.4f}, 정밀도:{1:.4f}, 재현율:{2:.4f}'.format(accuracy, precision, recall))

# In[10]:


lr_clf = LogisticRegression(solver='liblinear')#데이터양이 적은 경우 사용하는 solver

# In[16]:


pred

# In[15]:


lr_clf.fit(X_train,y_train)
pred = lr_clf.predict(X_test)
get_clf_eval(y_test, pred)

# In[17]:


#predict proba 에서 확인하면 
pred_proba = lr_clf.predict_proba(X_test)

# In[18]:


pred_proba

# In[19]:


# binarizer 얘를 통해 0과 1로 변환해서 반환할 예정
# theshold 지정하여 지정값보다 작거나 크면 0또는 1로 반환을 해준다.

from sklearn.preprocessing import Binarizer

# In[23]:


X= [[1,0,-1],
   [2,0,0],
   [0,1.3,1.5]]

binarizer=Binarizer(threshold=1.3)
# 1.3 기준으로 보면 2개 빼고 다 0

# In[24]:


print(binarizer.fit_transform(X))

# In[25]:


pred_proba[:,1].reshape(-1,1)

# In[28]:


tt_threshold = 0.5 # 임계값을 기준 0.5

pred_proba_1 =pred_proba[:,1].reshape(-1,1)
binarizer_tt=Binarizer(threshold=tt_threshold).fit(pred_proba_1)
tt_pred =binarizer_tt.transform(pred_proba_1)



# In[34]:


get_clf_eval(y_test, tt_pred)

# 오차 행렬
# [[110  25]
#  [ 43  37]]
# 정확도: 0.6837, 정밀도:0.5968, 재현율:0.4625

# In[32]:


tt_threshold = 0.4 # 임계값을 기준 0.4 기존 디폴트 0.5 -> 0.4로 변환

pred_proba_2 =pred_proba[:,1].reshape(-1,1)
binarizer_tt_1=Binarizer(threshold=tt_threshold).fit(pred_proba_2)
tt_pred_1 =binarizer_tt_1.transform(pred_proba_2)

get_clf_eval(y_test, tt_pred_1)

# In[33]:


tt_threshold = 0.6 # 임계값을 기준 0.4 기존 디폴트 0.5 -> 0.4로 변환

pred_proba_3 =pred_proba[:,1].reshape(-1,1)
binarizer_tt_2=Binarizer(threshold=tt_threshold).fit(pred_proba_3)
tt_pred_2 =binarizer_tt_2.transform(pred_proba_3)

get_clf_eval(y_test, tt_pred_2)

# - 0.4로 낮아지면서 -> 재현율:0.6250 재현율이 올라간 이유가 위의 FN, FP 값들의 변화가 잇으면서 발생한 것이 있다.
# - 줄어든 이유가 있다.
# 
# - 0.5-> 0.4 예측값이 많아지게 되다보니 -> 양성, 생존예측을 많이 하게 된다. 실제 양성을 음성 ( 실제 생존을 ,생존하지 않음) 예측하는 상대적으로 줄어들기 때문

# In[39]:


df_sp.survived.value_counts()

# In[38]:


## 임계값을 여러 가지 바꿔가면서 변화를 보자

thresholds= [0.3,0.4, 0.5, 0.6, 0.7, 0.8]

def get_eval_threshold(y_test, pred_proba_1, thresholds):
    for i in thresholds:
        binarizer=Binarizer(threshold=i).fit(pred_proba_1)
        pred =binarizer.transform(pred_proba_1)
        print('임계값의 따른 평가지표', i)
        get_clf_eval(y_test,pred)
        
get_eval_threshold(y_test,pred_proba[:,1].reshape(-1,1),thresholds )

# In[41]:


from sklearn.metrics import precision_recall_curve

#레이블 값이 1일때 예측확률을 추출
pred_proba_class1 = lr_clf.predict_proba(X_test)[:,1]

# 실제값 데이터 세트와 레이블 값이 1일 때의 예측확률을 precision_recall_curve 인자로 입력 
precisions, recalls, thresholds = precision_recall_curve(y_test, pred_proba_class1)
print('반환된 분류 결정 임곗값 배열의 shape:', thresholds.shape)

#반환된 임계값 배열 로우가 147건이므로 샘플로 10건만 추출하되, 임곗값을 15 step으로 추출
thr_index = np.arange(0,thresholds.shape[0],15)
print('샘플 추출을 위한 임계값 배열의 index 10개:', thr_index)
print('샘플용 10개의 임곗값:', np.round(thresholds[thr_index],2))

#15 step 단위로 추출된 임계값에 따른 정밀도와 재현율 값
print('샘플 임계값별 정밀도:', np.round(precisions[thr_index],3))
print('샘플 임계값별 재현율:', np.round(recalls[thr_index],3))

# In[43]:


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
%matplotlib inline

def precision_recall_curve_plot(y_test, pred_proba_c1):
    # thredshold ndarray와 이 threshold에 따른 정밀도, 재현율 ndarray 추출 
    precision, recalls, thresholds = precision_recall_curve(y_test, pred_proba_c1)
    
    #X축을 thredshold 값으로, Y축은 정밀도, 재현율 값으로 각각 plot 수행. 정밀도는 점선으로 표시 
    plt.figure(figsize=(8,6))
    threshold_boundary = thresholds.shape[0]
    plt.plot(thresholds, precision[0:threshold_boundary], linestyle='--', label='precision')
    plt.plot(thresholds, recalls[0:threshold_boundary], label ='recall')
    
    #thredshold 값 X축 Scale을 0.1단위로 변경
    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1),2))
    
    #X축, y축, label과 legend, 그리고 grid 설정
    plt.xlabel('Threshold value'); plt.ylabel('Precision and Recall value')
    plt.legend();plt.grid()
    plt.show()
    
precision_recall_curve_plot(y_test, lr_clf.predict_proba(X_test)[:,1])

# ###  정밀도 재현율에 대한 관계 
# - 생존에 대한 예측의 임계값이 변경에 따라서 정밀도, 재현율 수치가 변경이 된다. trade-off가 된다.
# - 단순하게 정확도나, 정밀도, 재현율, f1 하나만 보는 게 아니라 다 같이 종합적으로 바라봐야 한다.
# - 도메인이나, 어떤 비즈니스에 따라서 두 개 중 어떤 것을 좀 더 봐야 하는지 다를 수 있다. 
# - 정밀도나 재현율이 높아지는 경우
# 
# --- 
# - 정밀도가 높은 경우? 정밀도가 100%인 경우
# - 확실한 기준이 되는 경우만 생존으로 에측하고 나머지는 생존이 아닌 것으로 예측한다.
# - 예를 들어 여성이고, 1클래스에 있고 이런 것들의 사람들만 생존이라고 예측하고 나머지는 다 생존이 아니다라고 예측하는 것
# - 여기서 바라볼 수 있는 맹점
# - 정밀도 TP /(TP+FP)  확실한 생존한 사람만 생존이라고 예측하는 것이니 예측하는 절대치에 상관없이 1명만 제대로 예측하고 나머지는 생존하지 않았다 해도
# - 이 값은 100%가 나올 수 있다.
# - 1/(1+0)
# 
# --- 
# - 재현율로 바라보면? 재현율만 높은 경우 재현율 100%
# - TP /(TP+FN) 타이타닉데이터 기준으로 보면 전체를 다 생존으로 예측하는 것 실제 생존으로 예측한 사람 중에서 생존한 사람이 10명이라 해도 
# - 10/(10+0) 100%가 나올 수 있다.
# - 이 둘의 수치가 이러한 맹점이 있기 때문에 두 가지를 같이 잘 봐야하는 경우가 있고

# - F1스코어라는 것을 보는 이유도 이런 이슈가 있어서도 있다.
# - 재현율, 정밀도를 결합한 지표 

# In[45]:


from sklearn.metrics import f1_score

# In[51]:


#평가지표 함수
def get_clf_eval(y_test, pred):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    print('오차 행렬')
    print(confusion)
    print('정확도: {0:.4f}, 정밀도:{1:.4f}, 재현율:{2:.4f}, f1:{2:.4f}'.format(accuracy, precision, recall, f1))

# In[52]:


get_clf_eval(y_test, tt_pred_2)

# In[53]:


from sklearn.metrics import roc_curve

#레이블 값이 1일때의 예측 확률을 추출
pred_proba_class1 = lr_clf.predict_proba(X_test)[:,1]

fprs, tprs, thresholds = roc_curve(y_test, pred_proba_class1)
#반환된 임곗값 배열에서 샘플로 데이터를 추출하되, 임곗값을 5 step으로 추출
#thresholds[0]은 max(예측확률)+1로 임의 설정됨. 이를 제외하기 위해 np.arange는 1부터 시작 
thr_index = np.arange(1, thresholds.shape[0],5)

print('샘플 추출을 위한 임곗값 배열의 index:', thr_index)
print('샘플 index로 추출한 임곗값:', np.round(thresholds[thr_index],2))

# 5step 단위로 추출된 임계값에 따른 TPR, FPR 값
print('샘플 임곗값별 FPR:', np.round(fprs[thr_index], 3))
print('샘플 임곗값별 TPR:', np.round(tprs[thr_index], 3))

# In[54]:


def roc_curve_plot(y_test, pred_proba_c1):
    #임곗값에 따른 FPR, TPR값을 반환받음.
    fprs, tprs, thresholds = roc_curve(y_test, pred_proba_c1)
    #ROC 곡선을 그래프 곡선으로 그림
    plt.plot(fprs, tprs, label='ROC')
    #가운데 대각선 직선을 그림
    plt.plot([0,1], [0,1], 'k--', label='Random')
    
    #FPR X축의 Scale을 0.1 단위로 변경, X,Y축 명 설정 등
    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start,end,0.1),2))
    plt.xlim(0,1);plt.ylim(0,1)
    plt.xlabel('FPR(1-Specificity)');plt.ylabel('TPR(Recall)')
    plt.legend()
    
roc_curve_plot(y_test, pred_proba[:,1])

# In[56]:


from sklearn.metrics import roc_auc_score

pred_proba = lr_clf.predict_proba(X_test)[:,1]
roc_score = roc_auc_score(y_test, pred_proba)
print('ROC AUC 값:{0:.4f}'.format(roc_score))
