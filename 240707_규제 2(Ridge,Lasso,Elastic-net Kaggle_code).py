#!/usr/bin/env python
# coding: utf-8

# ### Kaggle code 탐색

# In[21]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from statsmodels.stats.diagnostic import normal_ad
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from scipy import stats
from scipy.special import inv_boxcox

# In[35]:


df = pd.read_csv('car data.csv')

# "Selling_Price" will be the dependent variable and the rest of the variables will be considered as independent variables.

# In[36]:


df

# In[37]:


df.describe(include='number')


# In[38]:


df.describe(include='object')


# Step 4.1: Feature Subset Selection

# In[39]:


df['Car_Name'].nunique()


# In[40]:


df.drop('Car_Name', axis=1, inplace=True)


# In[41]:


df.insert(0, "Age", df["Year"].max()+1-df["Year"] )
df.drop('Year', axis=1, inplace=True)
df.head()

# Step 4.3: Outlier Detection
# 

# In[42]:


sns.set_style('darkgrid')
colors = ['#0055ff', '#ff7000', '#23bf00']
CustomPalette = sns.set_palette(sns.color_palette(colors))

OrderedCols = np.concatenate([df.select_dtypes(exclude='object').columns.values, 
                              df.select_dtypes(include='object').columns.values])

fig, ax = plt.subplots(2, 4, figsize=(15,7),dpi=100)

for i,col in enumerate(OrderedCols):
    x = i//4
    y = i%4
    if i<5:
        sns.boxplot(data=df, y=col, ax=ax[x,y])
        ax[x,y].yaxis.label.set_size(15)
    else:
        sns.boxplot(data=df, x=col, y='Selling_Price', ax=ax[x,y])
        ax[x,y].xaxis.label.set_size(15)
        ax[x,y].yaxis.label.set_size(15)

plt.tight_layout()    
plt.show()

# As can be seen from the boxplots above, there are outliers in the dataset. We will identify the outliers based on the InterQuartile Range rule:

# In[43]:


outliers_indexes = []
target = 'Selling_Price'

for col in df.select_dtypes(include='object').columns:
    for cat in df[col].unique():
        df1 = df[df[col] == cat]
        q1 = df1[target].quantile(0.25)
        q3 = df1[target].quantile(0.75)
        iqr = q3-q1
        maximum = q3 + (1.5 * iqr)
        minimum = q1 - (1.5 * iqr)
        outlier_samples = df1[(df1[target] < minimum) | (df1[target] > maximum)]
        outliers_indexes.extend(outlier_samples.index.tolist())
        
        
for col in df.select_dtypes(exclude='object').columns:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3-q1
    maximum = q3 + (1.5 * iqr)
    minimum = q1 - (1.5 * iqr)
    outlier_samples = df[(df[col] < minimum) | (df[col] > maximum)]
    outliers_indexes.extend(outlier_samples.index.tolist())
    
outliers_indexes = list(set(outliers_indexes))
print('{} outliers were identified, whose indices are:\n\n{}'.format(len(outliers_indexes), outliers_indexes))

# It is not acceptable to drop an observation just because it is an outlier. They can be legitimate observations and it’s important to investigate the nature of the outlier before deciding whether to drop it or not. We are allowed to delete outliers in two cases:
# 
# Outlier is due to incorrectly entered or measured data
# Outlier creates a significant association

# In[44]:


# Outliers Labeling
df1 = df.copy()
df1['label'] = 'Normal'
df1.loc[outliers_indexes,'label'] = 'Outlier'

# Removing Outliers
removing_indexes = []
removing_indexes.extend(df1[df1[target]>33].index)
removing_indexes.extend(df1[df1['Kms_Driven']>400000].index)
df1.loc[removing_indexes,'label'] = 'Removing'

# Plot
target = 'Selling_Price'
features = df.columns.drop(target)
colors = ['#0055ff','#ff7000','#23bf00']
CustomPalette = sns.set_palette(sns.color_palette(colors))
fig, ax = plt.subplots(nrows=3 ,ncols=3, figsize=(15,12), dpi=200)

for i in range(len(features)):
    x=i//3
    y=i%3
    sns.scatterplot(data=df1, x=features[i], y=target, hue='label', ax=ax[x,y])
    ax[x,y].set_title('{} vs. {}'.format(target, features[i]), size = 15)
    ax[x,y].set_xlabel(features[i], size = 12)
    ax[x,y].set_ylabel(target, size = 12)
    ax[x,y].grid()

ax[2, 1].axis('off')
ax[2, 2].axis('off')
plt.tight_layout()
plt.show()

# In[45]:


removing_indexes = list(set(removing_indexes))
removing_indexes

# In[46]:


df.isnull().sum()

# Step 4.5: Discover Duplicates¶

# In[47]:


df[df.duplicated(keep=False)]

# Step 4.6: Drop Outliers
# 

# In[48]:


df1 = df.copy()
df1.drop(removing_indexes, inplace=True)
df1.reset_index(drop=True, inplace=True)

# We removed just two samples as outliers.
# 
# 

# Step 5: EDA
# 

# Step 5.1: Categorical Variables Univariate Analysis
# 

# In[49]:


CatCols = ['Fuel_Type', 'Seller_Type', 'Transmission']


# In[50]:


fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15,5), dpi=100)
colors = ['#0055ff', '#ff7000', '#23bf00']
CustomPalette = sns.set_palette(sns.color_palette(colors))
       
for i in range(len(CatCols)):
    graph = sns.countplot(x=CatCols[i], data=df1, ax=ax[i])
    ax[i].set_xlabel(CatCols[i], fontsize=15)
    ax[i].set_ylabel('Count', fontsize=12)
    ax[i].set_ylim([0,300])
    ax[i].set_xticklabels(ax[i].get_xticklabels(), fontsize=12)
    for cont in graph.containers:
        graph.bar_label(cont)

plt.suptitle('Frequency Distribution of Categorical Variables', fontsize=20) 
plt.tight_layout()
plt.show()

# Conclusion:
# There are 3 Fuel_Type categories. Petrol has the highest frequency and CNG has the least frequency.
# There are 2 Seller_Type categories. Dealer has the highest frequency and Individual has the least frequency.
# There are 2 Transmission categories. Manual has the highest frequency and Automatic has the least frequency.

# Step 5.2: Numerical Variables Univariate Analysis¶
# 

# In[51]:


NumCols = ['Age', 'Selling_Price', 'Present_Price', 'Kms_Driven', 'Owner']


# In[52]:


fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15,10), dpi=200)
c = '#0055ff'

for i in range(len(NumCols)):
    row = i//3
    col = i%3
    values, bin_edges = np.histogram(df1[NumCols[i]], 
                                     range=(np.floor(df1[NumCols[i]].min()), np.ceil(df1[NumCols[i]].max())))                
    graph = sns.histplot(data=df1, x=NumCols[i], bins=bin_edges, kde=True, ax=ax[row,col],
                         edgecolor='none', color=c, alpha=0.4, line_kws={'lw': 2.5})
    ax[row,col].set_xlabel(NumCols[i], fontsize=15)
    ax[row,col].set_ylabel('Count', fontsize=12)
    ax[row,col].set_xticks(np.round(bin_edges,1))
    ax[row,col].set_xticklabels(ax[row,col].get_xticks(), rotation = 45)
    ax[row,col].grid(color='lightgrey')
    for j,p in enumerate(graph.patches):
        ax[row,col].annotate('{}'.format(p.get_height()), (p.get_x()+p.get_width()/2, p.get_height()+1),
                             ha='center', fontsize=10 ,fontweight="bold")
    
    textstr = '\n'.join((
    r'$\mu=%.2f$' %df1[NumCols[i]].mean(),
    r'$\sigma=%.2f$' %df1[NumCols[i]].std(),
    r'$\mathrm{median}=%.2f$' %np.median(df1[NumCols[i]]),
    r'$\mathrm{min}=%.2f$' %df1[NumCols[i]].min(),
    r'$\mathrm{max}=%.2f$' %df1[NumCols[i]].max()
    ))
    ax[row,col].text(0.6, 0.9, textstr, transform=ax[row,col].transAxes, fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round',facecolor='#509aff', edgecolor='black', pad=0.5))

ax[1, 2].axis('off')
plt.suptitle('Distribution of Numerical Variables', fontsize=20) 
plt.tight_layout()   
plt.show()


# Step 5.3: Target vs. Numerical Features Bivariate Analysis
# 
# Plot Selling_Price vs. numerical features:
# 
# 

# In[53]:


fig, ax = plt.subplots(nrows=2 ,ncols=2, figsize=(10,10), dpi=90)
num_features = ['Present_Price', 'Kms_Driven', 'Age', 'Owner']
target = 'Selling_Price'
c = '#0055ff'

for i in range(len(num_features)):
    row = i//2
    col = i%2
    ax[row,col].scatter(df1[num_features[i]], df1[target], color=c, edgecolors='w', linewidths=0.25)
    ax[row,col].set_title('{} vs. {}'.format(target, num_features[i]), size = 12)
    ax[row,col].set_xlabel(num_features[i], size = 12)
    ax[row,col].set_ylabel(target, size = 12)
    ax[row,col].grid()

plt.suptitle('Selling Price vs. Numerical Features', size = 20)
plt.tight_layout()
plt.show()

# Conclusion:
# As Present_Price increases, Selling_Price increases as well. So Selling_Price is directly proportional to Present_Price.
# 
# As the car's Kms_Driven increases, its Selling_Price decreases. So Selling_Price is inversely proportional to Kms_Driven.
# 
# As the car ages, its Selling_Price decreases. So Selling_Price is inversely proportional to the Age of the car.
# 
# As the number of previous car owners increases, its Selling_Price decreases. So Selling_Price is inversely proportional to Owner.

# Step 5.4: Target vs. Categorical Features Bivariate Analysis¶
# 
# Selling_Price vs. categorical features strip plots:
# 
# 

# In[54]:


fig, axes = plt.subplots(nrows=1 ,ncols=3, figsize=(12,5), dpi=100)
cat_features = ['Fuel_Type', 'Seller_Type', 'Transmission']
target = 'Selling_Price'
c = '#0055ff'

for i in range(len(cat_features)):
    sns.stripplot(ax=axes[i], x=cat_features[i], y=target, data=df1, size=6, color=c)
    axes[i].set_title('{} vs. {}'.format(target, cat_features[i]), size = 13)
    axes[i].set_xlabel(cat_features[i], size = 12)
    axes[i].set_ylabel(target, size = 12)
    axes[i].grid()

plt.suptitle('Selling Price vs. Categorical Features', size = 20)
plt.tight_layout()
plt.show()

# Conclusion:
# Diesel Cars > CNG Cars > Petrol Cars in terms of Selling_Price.
# The Selling_Price of cars sold by individuals is lower than the price of cars sold by dealers.
# Automatic cars are more expensive than manual c

# Step 5.5: Multivariate Analysis¶
# 

# In[55]:


graph = sns.lmplot(x='Present_Price', y='Selling_Price', data= df1, fit_reg=False, row='Seller_Type',
                   col='Transmission', hue='Fuel_Type', palette=CustomPalette, height=4, aspect=1)   

plt.suptitle('Selling_Price vs. Present_Price', fontsize=20) 
sns.move_legend(graph, "lower center", bbox_to_anchor=(1.05, 0.5), ncol=1)
plt.tight_layout()
plt.show()

# Conclusion:
# All of the Individual Seller_Type have had Petrol cars.
# Diesel cars all have belonged to the Dealer Seller_Type.
# All of the CNG cars have had Manual Transmission and have belonged to Dealer Seller_Type.

# Step 6: Categorical Variables Encoding
# We implement dummy encoding on categorical columns, since they are all nominal variables:

# In[56]:


CatCols = ['Fuel_Type', 'Seller_Type', 'Transmission']

df1 = pd.get_dummies(df1, columns=CatCols, drop_first=True)
df1.head(5)

# Step 7: Correlation Analysis¶
# 

# In[57]:


target = 'Selling_Price'
cmap = sns.diverging_palette(125, 28, s=100, l=65, sep=50, as_cmap=True)
fig, ax = plt.subplots(figsize=(9, 8), dpi=80)
ax = sns.heatmap(pd.concat([df1.drop(target,axis=1), df1[target]],axis=1).corr(), annot=True, cmap=cmap)
plt.show()

# The target variable "Selling Price" is highly correlated with Present_Price & Seller_Type & Fuel_Type.
# Some independent variables like Fuel_Type_Petrol and Fuel_Type_Disel are highly correlated, which is called Multicollinearity.

# Step 8: Build Linear Regression Model¶
# 
# Step 8.1: Determine Features & Target Variables¶
# 

# In[58]:


X = df1.drop('Selling_Price', axis=1)
y = df1['Selling_Price']

# In[59]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[60]:


print('X_train shape: ', X_train.shape)
print('X_test shape: ', X_test.shape)
print('y_train shape: ', y_train.shape)
print('y_test shape: ',y_test.shape)

# In[61]:


y_test_actual = y_test


# Step 8.3: Scale Data using Standard Scaler
# 

# In[62]:


scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# It is very important that StandardScaler transformation should only be learnt from the training set, otherwise it will lead to data leakage.
# 
# 

# Step 8.4: Train the Model¶
# 

# In[64]:


linear_reg = LinearRegression()
linear_reg.fit(X_train_scaled, y_train)

# In[65]:


pd.DataFrame(data = np.append(linear_reg.intercept_ , linear_reg.coef_), 
             index = ['Intercept']+[col+" Coef." for col in X.columns], columns=['Value']).sort_values('Value', ascending=False)

# Step 8.5: Model Evaluation
# 

# In[66]:


def model_evaluation(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    
    MAE = metrics.mean_absolute_error(y_test, y_pred)
    MSE = metrics.mean_squared_error(y_test, y_pred)
    RMSE = np.sqrt(MSE)
    R2_Score = metrics.r2_score(y_test, y_pred)
    
    return pd.DataFrame([MAE, MSE, RMSE, R2_Score], index=['MAE', 'MSE', 'RMSE' ,'R2-Score'], columns=[model_name])

# In[67]:


model_evaluation(linear_reg, X_test_scaled, y_test, 'Linear Reg.')


# Step 8.6: Model Evaluation using Cross-Validation¶
# By using cross-validation, we can have more confidence in our estimation for the model evaluation metrics than the former simple train-test split:

# In[68]:


linear_reg_cv = LinearRegression()
scaler = StandardScaler()
pipeline = make_pipeline(StandardScaler(),  LinearRegression())

kf = KFold(n_splits=6, shuffle=True, random_state=0) 
scoring = ['neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_root_mean_squared_error', 'r2']
result = cross_validate(pipeline, X, y, cv=kf, return_train_score=True, scoring=scoring)

MAE_mean = (-result['test_neg_mean_absolute_error']).mean()
MAE_std = (-result['test_neg_mean_absolute_error']).std()
MSE_mean = (-result['test_neg_mean_squared_error']).mean()
MSE_std = (-result['test_neg_mean_squared_error']).std()
RMSE_mean = (-result['test_neg_root_mean_squared_error']).mean()
RMSE_std = (-result['test_neg_root_mean_squared_error']).std()
R2_Score_mean = result['test_r2'].mean()
R2_Score_std = result['test_r2'].std()

pd.DataFrame({'Mean': [MAE_mean,MSE_mean,RMSE_mean,R2_Score_mean], 'Std': [MAE_std,MSE_std,RMSE_std,R2_Score_std]},
             index=['MAE', 'MSE', 'RMSE' ,'R2-Score'])

# The linear regression model obtained R2-score of %85.57 using 6-fold cross-validation.
# 
# Pipeline is a great way to prevent data leakage as it ensures that the appropriate method is performed on the correct data subset. This is ideal for using in cross-validation since it ensures that only the training folds are used when performing fit and the test set (validation set) is used only for calculating the accuracy score in each iteration of cross-validation.

# Step 8.7: Assumptions
# We need to check the assumptions of linear regression, because if the assumptions are not met, the interpretation of the results will not always be valid:

# Step 8.7.1: Assumption 1 - Linearity
# This assumes that there is a linear relationship between the independent variables or features and the dependent variable or label. Fitting a linear model to data with non-linear patterns results in serious prediction errors, because our model is underfitting.

# To detect nonlinearity, we can check:
# 
# Plots of actual vs. predicted values -> The desired outcome is that points are symmetrically distributed around a diagonal line
# Plots of residuals vs. predicted values -> The desired outcome is that points are symmetrically distributed around a horizontal line
# In both cases we should have an almost constant variance.

# In[69]:


def residuals(model, X_test, y_test):
    '''
    Creates predictions on the features with the model and calculates residuals
    '''
    y_pred = model.predict(X_test)
    df_results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    df_results['Residuals'] = abs(df_results['Actual']) - abs(df_results['Predicted'])
    
    return df_results

# In[70]:


def linear_assumption(model, X_test, y_test):
    '''
    Function for visually inspecting the assumption of linearity in a linear regression model
    '''
    df_results = residuals(model, X_test, y_test)
    
    fig, ax = plt.subplots(1,2, figsize=(15,6), dpi=80)
    sns.regplot(x='Predicted', y='Actual', data=df_results, lowess=True, ax=ax[0],
                color='#0055ff', line_kws={'color':'#ff7000','ls':'--','lw':2.5})
    ax[0].set_title('Actual vs. Predicted Values', fontsize=15)
    ax[0].set_xlabel('Predicted', fontsize=12)
    ax[0].set_ylabel('Actual', fontsize=12)        
    
    sns.regplot(x='Predicted', y='Residuals', data=df_results, lowess=True, ax=ax[1],
                color='#0055ff', line_kws={'color':'#ff7000','ls':'--','lw':2.5})
    ax[1].set_title('Residuals vs. Predicted Values', fontsize=15)
    ax[1].set_xlabel('Predicted', fontsize=12)
    ax[1].set_ylabel('Residuals', fontsize=12)  

# In[71]:


linear_assumption(linear_reg, X_test_scaled, y_test)

# The inspection of the plots shows that the linearity assumption is not satisfied.
# 
# 

# Potential solutions:
# 
# Applying nonlinear transformations
# Adding polynomial terms to some of the predictors

# Step 8.7.2: Assumption 2 - Normality of Residuals
# This assumes that the error terms of the model are normally distributed with a mean value of zero.
# 
# This can actually happen if either the predictors or the label are significantly non-normal. Other potential reasons could include the linearity assumption being violated or presence of a few large outliers in data affecting our model.
# 
# A violation of this assumption could cause issues with either shrinking or inflating our confidence intervals. When the residuals distribution significantly departs from Gaussian, confidence intervals may be too wide or too narrow. Technically, we can omit this assumption if we assume instead that the model equation is correct and our goal is to estimate the coefficients and generate predictions (in the sense of minimizing mean squared error). However, normally we are interested in making valid inferences from the model or estimating the probability that a given prediction error will exceed some threshold in a particular direction. To do so, the assumption about the normality of residuals must be satisfied.
# 
# To investigate this assumption we can check:
# 
# Check residuals histogram
# Quantile-Quantile probability plot -> plotting the residuals vs the order of statistic
# Anderson-Darling test

# In[72]:


def normal_errors_assumption(model, X_test, y_test, p_value_thresh=0.05):
    '''
    Function for inspecting the assumption of normality of residuals.
    '''
    df_results = residuals(model, X_test, y_test)
    
    # Anderson-Darling Test
    p_value = normal_ad(df_results['Residuals'])[1]
    
    print('\nP-value from the test (below 0.05 generally means non-normal):  ', np.round(p_value,6))
    if p_value < p_value_thresh:
        print('Residuals are not normally distributed. Assumption not satisfied.') 
    else:
        print('Residuals are normally distributed. Assumption satisfied.')

      
    # Residuals Histogram
    fig, ax = plt.subplots(1,2, figsize=(15,6), dpi=80)
    
    sns.histplot(data=df_results, x='Residuals', kde=True, ax=ax[0], bins=15, 
                 color='#0055ff', edgecolor='none', alpha=0.4, line_kws={'lw': 2.5})
    ax[0].set_xlabel('Residuals', fontsize=12)
    ax[0].set_ylabel('Count', fontsize=12)
    ax[0].set_title('Distribution of Residuals', fontsize=15)
    textstr = '\n'.join((
        r'$\mu=%.2f$' %np.mean(df_results['Residuals']),
        r'$\sigma=%.2f$' %np.std(df_results['Residuals']),
        ))
    ax[0].text(0.7, 0.9, textstr, transform=ax[0].transAxes, fontsize=15, verticalalignment='top',
                 bbox=dict(boxstyle='round',facecolor='#509aff', edgecolor='black', pad=0.5))
    
    
    # Q-Q Probability Plot
    stats.probplot(df_results['Residuals'], dist="norm", plot= ax[1])
    ax[1].set_title("Residuals Q-Q Plot", fontsize=15)
    ax[1].set_xlabel('Theoretical Quantiles', fontsize=12)
    ax[1].set_ylabel('Ordered Values', fontsize=12)
    ax[1].get_lines()[0].set_markerfacecolor('#509aff')
    ax[1].get_lines()[1].set_color('#ff7000')
    ax[1].get_lines()[1].set_linewidth(2.5)
    ax[1].get_lines()[1].set_linestyle('--')
    ax[1].legend(['Actual','Theoretical'])
    
    plt.show()

# In[73]:


normal_errors_assumption(linear_reg, X_test_scaled, y_test)


#  QQ Plot of residuals:
# 
# The bow-shaped pattern of deviations from the diagonal implies that the residuals have excessive skewness.
# The s-shaped pattern of deviations from the diagonal implies excessive kurtosis of the residuals (there are either too many or too few large errors in both directions.)
# The non-zero mean value and the positive skewness of the residual distribution and the s-shaped pattern of the deviations in the QQ plot show that the residuals do not follow the Gaussian distribution.
# 
# Potential solutions:
# 
# Nonlinear transformation of target variable and features
# Removing potential outliers

# Step 8.7.3: Assumption 3 - No Perfect Multicollinearity
# Multicollinearity occurs when the independent variables are correlated to each other. It becomes difficult for the model to estimate the relationship between each independent variable and the dependent variable independently because the independent variables tend to change in unison. The coefficient estimates can swing wildly based on which other independent variables are in the model and they become very sensitive to small changes in the model. Therefore, the estimates will be less precise and highly sensitive to particular sets of data. This increases the standard error of the coefficients, which results in them potentially showing as statistically insignificant when they might actually be significant. On the other hand, the simultaneous changes of the independent variables can lead to large fluctuations of the target variable, which leads to the overfitting of the model and the reduction of its performance.
# 
# To detect multicolinearity, we can:
# 
# Use a heatmap of the correlation (step 7)
# Examine the variance inflation factor (VIF)
# Interpretation of VIF: The square root of a given variable’s VIF shows how much larger the standard error is, compared with what it would be if that predictor were uncorrelated with the other features in the model. The higher the value of VIF the higher correlation between this variable and the rest. A rule of thumb is that if VIF > 10 then multicollinearity is high.

# In[82]:


X['Fuel_Type_Diesel'] = X['Fuel_Type_Diesel']*1
X['Fuel_Type_Petrol'] = X['Fuel_Type_Petrol']*1
X['Seller_Type_Individual'] = X['Seller_Type_Individual']*1
X['Transmission_Manual'] = X['Transmission_Manual']*1

# In[83]:


def multicollinearity_assumption(X):
    ''''
    This assumes that predictors are not correlated with each other and calculates VIF values of predictors
    '''
    vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    
    possible_multicollinearity = sum([1 for v in vif if v > 10])
    definite_multicollinearity = sum([1 for v in vif if v > 100])
    
    print('{} cases of possible multicollinearity.'.format(possible_multicollinearity))
    print('{} cases of definite multicollinearity.'.format(definite_multicollinearity))
    
    if definite_multicollinearity == 0:
        if possible_multicollinearity == 0:
            print('Assumption satisfied.')
        else:
            print('Assumption possibly satisfied.')
    else:
        print('Assumption not satisfied.')
    
    return pd.DataFrame({'vif': vif}, index=X.columns).round(2)

# In[84]:


multicollinearity_assumption(X)


# There is possible multicollinearity for Fuel_Type_Petrol.
# 
# 

# Potential solutions:
# 
# Using Regularization
# Removing features with high values of VIF
# Using PCA -> Reducing features to a smaller set of uncorrelated components

# Step 8.7.4: Assumption 4 - No Autocorrelation of Residuals¶
# 
# 
# This assumes no autocorrelation of the residuals. The presence of autocorrelation usually indicates that we are missing some information that should be captured by the model. Our model can be systematically biased by under-prediction or over-prediction under certain conditions. This could be the result of violating the linearity assumption.
# 
# To investigate this assumption we can perform a Durbin-Watson test to determine whether the correlation is positive or negative:
# 
# The test statistic always has a value between 0 and 4
# Values of 1.5 < d < 2.5 means that there is no autocorrelation in the data
# Values < 1.5 indicate positive autocorrelation, values > 2.5 indicate negative autocorrelation

# In[85]:


def autocorrelation_assumption(model, X_test, y_test):
    '''
    It assumes that there is no autocorrelation in the residuals. If there is autocorrelation, then 
    there is a pattern that is not explained because the current value is dependent on the previous value.
    '''
    df_results = residuals(model, X_test, y_test)

    durbinWatson = durbin_watson(df_results['Residuals'])
    print('Durbin-Watson:', round(durbinWatson,3))
    
    if durbinWatson < 1.5:
        print('Signs of positive autocorrelation. Assumption not satisfied.', '\n')
    elif durbinWatson > 2.5:
        print('Signs of negative autocorrelation. Assumption not satisfied.', '\n')
    else:
        print('Little to no autocorrelation. Assumption satisfied.', '\n')

# In[86]:


autocorrelation_assumption(linear_reg, X_test_scaled, y_test)


# Durbin-Watson test indicates there is no autocorrelation.
# 
# Potential solution for fixing autocorrelation of residuals:
# 
# Adding interaction terms

# Step 8.7.5: Assumption 5 - Homoscedasticity
# Homoscedasticity means that the residuals doesn’t change across all the values of the target variable.
# 
# When residuals do not have constant variance, it is difficult to determine the true standard deviation of the forecast errors, usually resulting in confidence intervals that are too wide/narrow. The effect of heteroscedasticity might also be putting too much weight to a subset of data when estimating coefficients.
# 
# To investigate if the residuals are homoscedastic, we can look at a plot of residuals vs. predicted values. The placement of the points should be random and no pattern (increase/decrease in values of residuals) should be visible.

# In[87]:


def homoscedasticity_assumption(model, X_test, y_test):
    """
    Homoscedasticity assumes that the residuals exhibit constant variance
    """
    print('The orange line should be flat:')
    df_results = residuals(model, X_test, y_test)
    
    fig = plt.figure(figsize=(6,6), dpi=80)
    sns.regplot(x='Predicted', y='Residuals', data=df_results, lowess=True,
                color='#0055ff', line_kws={'color':'#ff7000','ls':'--','lw':2.5})
    plt.axhline(y=0, color='#23bf00', lw=1)
    plt.title('Residuals vs. Predicted Values', fontsize=15)
    plt.xlabel('Predicted Values', fontsize=12)
    plt.ylabel('Residuals', fontsize=12)    
    plt.show()

# In[88]:


homoscedasticity_assumption(linear_reg, X_test_scaled, y_test)


# We can not see a fully uniform variance across our residuals because the orange line is not flat. The assumption is not satisfied.
# 
# 
# Potential solutions:
# 
# Outlier removal
# Appllying log transformation of independent variables
# Applying polynomial regression
# 
# 
# In the following:
# To satisfy the multicollinearity assumption, we remove the Fuel_Type_Petrol feature.
# Then, according to the pattern in the plot of the residuals vs. predicted values, we will use box-cox transformation on the entire dataset.
# By applying polynomial regression, we will try to improve the satisfaction of homoscedasticity and normality of residuals.
# Finally, we will use regularization to reduce the probability of the model to be overfit.
# Drop Fuel_Type_Petrol:

# In[89]:


del df1['Fuel_Type_Petrol']


# Step 8.8: Results Visualization
# We compare the actual and predicted target values for the test data with the help of a bar plot:

# In[90]:


y_test_pred = linear_reg.predict(X_test_scaled)
df_comp = pd.DataFrame({'Actual':y_test_actual, 'Predicted':y_test_pred})

# In[91]:


def compare_plot(df_comp):
    df_comp.reset_index(inplace=True)
    df_comp.plot(y=['Actual','Predicted'], kind='bar', figsize=(20,7), width=0.8)
    plt.title('Predicted vs. Actual Target Values for Test Data', fontsize=20)
    plt.ylabel('Selling_Price', fontsize=15)
    plt.show()

# In[92]:


compare_plot(df_comp)


# The difference between the corresponding bars in the above bar plot shows the prediction error of the model on the test data. Also, out of 90 test samples, Selling_Price has been predicted negatively in 6 cases. A negative prediction for Selling_Price is disappointing.

# Step 9: Apply Box-Cox Transformation:
# 

# In order to satisfy the regression assumptions, we apply the Box-Cox transformation on the whole dataset. The Box-Cox transformations change the shape of our data, making it more close to a normal distribution.
# 
# In order to prevent data leakage, the fitted lambda value for each feature is obtained from the training data set and then the transformation is applied to the both training and test data:
# 
# 

# Step 9.1: Transform Training Data & Save Lambda Values¶
# 

# In[93]:


fitted_lambda = pd.Series(np.zeros(len(df1.columns), dtype=np.float64), index=df1.columns)

y_train, fitted_lambda['Selling_Price'] = stats.boxcox(y_train+1)
for col in X_train.columns:
    X_train[col], fitted_lambda[col] = stats.boxcox(X_train[col]+1)
    
fitted_lambda

# Step 9.2: Transform Test Data Using Lambda Values¶
# 

# In[94]:


y_test = stats.boxcox(y_test+1, fitted_lambda['Selling_Price'])
for col in X_test.columns:
    X_test[col] = stats.boxcox(X_test[col]+1, fitted_lambda[col])

# In[95]:


y_train = pd.DataFrame(y_train, index=X_train.index, columns=['Selling_Price'])
y_test = pd.DataFrame(y_test, index=X_test.index, columns=['Selling_Price'])

X_boxcox = pd.concat([X_train, X_test])
y_boxcox = pd.concat([y_train, y_test])

df_boxcox = pd.concat([X_boxcox, y_boxcox], axis=1)
df_boxcox.sort_index(inplace=True)

del df_boxcox['Fuel_Type_Petrol']

# As seen in Step 5.2, the distribution of continuous variables all had a lot of positive skewness. In the following, we can see the change in the shape of the distribution of these variables after applying Box-Cox transformation:

# In[96]:


fig, ax = plt.subplots(2, 4, figsize=(15,8), dpi=100)
columns = ['Selling_Price', 'Present_Price', 'Kms_Driven', 'Age']

for i,col in enumerate(columns):
    sns.kdeplot(df1[col], label="Non-Normal", fill=True, color='#0055ff', linewidth=2, ax=ax[0,i])
    sns.kdeplot(df_boxcox[col], label="Normal", fill=True, color='#23bf00', linewidth=2, ax=ax[1,i])  
    ax[0,i].set_xlabel('', fontsize=15)
    ax[1,i].set_xlabel(col, fontsize=15, fontweight='bold')
    ax[0,i].legend(loc="upper right")
    ax[1,i].legend(loc="upper right")

ax[0,2].tick_params(axis='x', labelrotation = 20)
plt.suptitle('Data Transformation using Box-Cox', fontsize=20)
plt.tight_layout()
plt.show()

# As can be seen, the distribution of the variables is closer to the normal distribution after applying Box-Cox.
# 
# 

# In[97]:


X = df_boxcox.drop('Selling_Price', axis=1)
y = df_boxcox['Selling_Price']

# Now box-cox transformation is applied on X and y. Next, we will add higher order features to the model.
# 
# 

# Step 10: Build 2nd-order Polynomial Regression¶
# 

# In order to overcome under-fitting and meet linear regression assumptions, first we try to increase the complexity of the model by adding all of the second-order terms of the features

# Step 10.1: Create 2nd-order Polynomial Features¶
# 

# In[98]:


poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
X_poly = pd.DataFrame(X_poly, columns=poly_features.get_feature_names_out(X.columns))
X_poly.head(5)

# In[99]:


poly_features_names = poly_features.get_feature_names_out(X.columns)
len(poly_features_names)

# Step 10.2: Split Dataset to Training & Test Sets¶
# 

# In[100]:


X_poly_train, X_poly_test, y_poly_train, y_poly_test = train_test_split(X_poly, y, test_size=0.3, random_state=0)

# In[101]:


print('X_poly_train shape: ', X_poly_train.shape)
print('X_poly_test shape: ', X_poly_test.shape)
print('y_poly_train shape: ', y_poly_train.shape)
print('y_poly_test shape: ',y_poly_test.shape)

# Step 10.3: Scale Data using Standard Scaler¶
# 

# In[102]:


scaler = StandardScaler()
scaler.fit(X_poly_train)

X_poly_train = scaler.transform(X_poly_train)
X_poly_train = pd.DataFrame(X_poly_train, columns=poly_features_names)

X_poly_test = scaler.transform(X_poly_test)
X_poly_test = pd.DataFrame(X_poly_test, columns=poly_features_names)

# Step 10.4: Create Polynomial Regression Model using Linear Regression¶
# 

# In[103]:


polynomial_reg = LinearRegression()
polynomial_reg.fit(X_poly_train, y_poly_train)

# Step 10.5: Model Evaluation
# 

# 2nd-order Polynomial Model Performance on Test Data:
# 
# 

# In[104]:


model_evaluation(polynomial_reg, X_poly_test, y_poly_test, 'Polynomial Reg. Test')


# 2nd-order Polynomial Model Performance on Training Data:
# 
# 

# In[105]:


model_evaluation(polynomial_reg, X_poly_train, y_poly_train, 'Polynomial Reg. Train')


# As can be seen, using boxcox transformation and production of second-order features has improved the model performance greatly!
# 
# MAE: 1.199 -> 0.088
# 
# MSE: 3.715 -> 0.011
# 
# RMSE: 1.927 -> 0.107
# 
# R2 Score: %88.72 -> %98.16
# 
# The accuracy of the model on the training and test data are close to each other, so the model is not overfit.
# The production of third-order and higher features was also tested, which caused the model to become overfit.

# Step 10.6: Model Evaluation using Cross-Validation
# By using cross-validation, we can be more confident in our estimation of the model evaluation metrics:

# In[106]:


pipeline = make_pipeline(PolynomialFeatures(degree=2), StandardScaler(),  LinearRegression())

kf = KFold(n_splits=6, shuffle=True, random_state=0) 
scoring = ['neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_root_mean_squared_error', 'r2']
result2 = cross_validate(pipeline, X, y, cv=kf, return_train_score=True, scoring=scoring)

MAE_mean = (-result2['test_neg_mean_absolute_error']).mean()
MAE_std = (-result2['test_neg_mean_absolute_error']).std()
MSE_mean = (-result2['test_neg_mean_squared_error']).mean()
MSE_std = (-result2['test_neg_mean_squared_error']).std()
RMSE_mean = (-result2['test_neg_root_mean_squared_error']).mean()
RMSE_std = (-result2['test_neg_root_mean_squared_error']).std()
R2_Score_mean = result2['test_r2'].mean()
R2_Score_std = result2['test_r2'].std()

pd.DataFrame({'Mean': [MAE_mean,MSE_mean,RMSE_mean,R2_Score_mean], 'Std': [MAE_std,MSE_std,RMSE_std,R2_Score_std]},
             index=['MAE', 'MSE', 'RMSE' ,'R2-Score'])

# The accuracy obtained from the simple train test split is valid because it is close to the accuracy obtained from cross validation.
# 
# So far, we have a polynomial model consisting of 35 features with 98.16% r2-score. Lets check regression assumptions for this model.

# Step 10.7: Assumptions Investigation
# Step 10.7.1: Assumption 1 - Linearity

# In[107]:


linear_assumption(polynomial_reg, X_poly_test, y_poly_test)


# Step 10.7.2: Assumption 2 - Normality of Residuals¶
# 

# In[108]:


normal_errors_assumption(polynomial_reg, X_poly_test, y_poly_test)


# Step 10.7.3: Assumption 3 - No Perfect Multicollinearity¶
# 

# In[110]:


warnings.simplefilter(action='ignore')
multicollinearity_assumption(X_poly).T

# Step 10.7.4: Assumption 4 - No Autocorrelation of Residuals¶
# 

# In[111]:


autocorrelation_assumption(polynomial_reg, X_poly_test, y_poly_test)


# Step 10.7.5: Assumption 5 - Homoscedasticity¶
# 

# In[112]:


homoscedasticity_assumption(polynomial_reg, X_poly_test, y_poly_test)


# All regression assumptions are satisfied to a good extent except for multicollinearity. But in general, the polynomial model performs better than the linear model on this data set.

# To reduce multicollinearity effects we can use Regularization. Regularized regression puts contraints on the magnitude of the coefficients and will progressively shrink them towards zero relative to the least-squares estimates. There are two types of regularization as follows:
# 
# L1 Regularization or Lasso Regularization
# L2 Regularization or Ridge Regularization

# Step 11: Ridge Regression
# In L2 Regularization or Ridge Regularization, we add a penalty which is the sum of the squared values of weights on the loss function in order to push the estimated coefficients towards zero and not take more extreme values:
# 
# Loss function = OLS + alpha * (the sum of the square of coefficients)
# In the above loss function, alpha is the tuning factor which has control over the strength of the penalty term. A small alpha value leads the model to overfit and a large alpha value leads the model to underfit. We use cross-validated ridge regression to tune alpha value:

# Step 11.1: Split Dataset to Training & Test Sets¶
# 

# In[113]:


X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.3, random_state=0)


# Step 11.2: Scale Data using Standard Scaler¶
# 

# In[114]:


scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Step 11.3: Build Ridge Regression Model¶
# 

# In[115]:


alphas = 10**np.linspace(10,-2,100)*0.5

ridge_cv_model = RidgeCV(alphas = alphas, cv = 3, scoring = 'neg_mean_squared_error')                        
ridge_cv_model.fit(X_train, y_train)

# The alpha value that results in the smallest cross-validation MSE is:
# 
# 

# In[116]:


ridge_cv_model.alpha_


# Step 11.4: Ridge Model Evaluation
# 

# In[117]:


model_evaluation(ridge_cv_model, X_test, y_test, 'Ridge Reg. Test')


# Ridge Model Performance on Training Data:
# 
# 

# In[118]:


model_evaluation(ridge_cv_model, X_train, y_train, 'Ridge Reg. Train')


# The accuracy of the model on the training and test data are close to each other, so the model is not overfit.
# The Ridge model has almost the same accuracy as the polynomial model.

# Step 11.5: Ridge Regression Coefficients¶
# 

# The Ridge Regression coefficients are:
# 
# 

# In[119]:


ridge_cv_model.coef_


# As can be seen, L2 regularization allows weights to decay towards zero but not to zero.
# 
# 

# Step 11.6: Ridge Model Assumptions Investigation
# 

# Step 11.6.1: Assumption 1 - Linearity¶
# 

# In[120]:


linear_assumption(ridge_cv_model, X_test, y_test)


# Step 11.6.2: Assumption 2 - Normality of Residuals
# 

# In[122]:


normal_errors_assumption(ridge_cv_model, X_test, y_test)


# In[123]:


multicollinearity_assumption(X_poly).T


# In[124]:


autocorrelation_assumption(ridge_cv_model, X_test, y_test)


# In[125]:


homoscedasticity_assumption(ridge_cv_model, X_test, y_test)


# As can be seen, the assumptions have improved slightly compared to the polynomial model.¶
# 

# Step 12: Lasso Regression
# 

# In L1 Regularization or Lasso Regularization, we add a penalty which is the sum of the absolute values of weights on the loss function in order to push the estimated coefficients towards zero:
# 
# Loss function = OLS + alpha * (the sum of the absolute of coefficients)
# We again use cross-validated lasso regression to tune alpha value:
# 
# 

# Step 12.1: Build Lasso Regression Model¶
# 

# In[126]:


lasso_cv_model = LassoCV(eps=0.01, n_alphas=100, max_iter=10000, cv=3)


# In[127]:


lasso_cv_model.fit(X_train, y_train)


# In[128]:


lasso_cv_model.alpha_


# Step 12.2: Lasso Model Evaluation on Test Data
# 

# In[129]:


model_evaluation(lasso_cv_model, X_test, y_test, 'Lasso Reg. Test')


# In[130]:


model_evaluation(lasso_cv_model, X_train, y_train, 'Lasso Reg. Train')


# The accuracy of the model on the training and test data are close to each other, so the model is not overfit.
# The accuracy of the Lasso model has decreased slightly compared to the ridge model.

# Step 12.3: Lasso Regression Coefficients
# 

# In[131]:


lasso_coef = lasso_cv_model.coef_
lasso_coef

# In[132]:


lasso_coef = pd.DataFrame(lasso_cv_model.coef_, index=X_poly.columns, columns=['Lasso Coef.'])
lasso_coef = lasso_coef[lasso_coef['Lasso Coef.']!=0]
lasso_coef.T

# L1 regularization allows weights to decay to zero.
# In exchange for reducing the number of features from 35 to 15, the R2-score of the model has decreased from %98.16 to %96.61.

# Step 12.4: Lasso Model Assumptions Investigation
# Step 12.4.1: Assumption 1 - Linearity

# In[133]:


linear_assumption(lasso_cv_model, X_test, y_test)


# Step 12.4.2: Assumption 2 - Normality of Residuals¶
# 

# In[134]:


normal_errors_assumption(lasso_cv_model, X_test, y_test)


# Step 12.4.3: Assumption 3 - No Perfect Multicollinearity
# 

# In[135]:


multicollinearity_assumption(X_poly[lasso_coef.index]).T


# Step 12.4.4: Assumption 4 - No Autocorrelation of Residuals¶
# 

# In[136]:


autocorrelation_assumption(lasso_cv_model, X_test, y_test)


# Step 12.4.5: Assumption 5 - Homoscedasticity¶
# 

# In[137]:


homoscedasticity_assumption(lasso_cv_model, X_test, y_test)


# Step 13: Elastic-Net Regression¶
# Elastic-Net allows a balance of both L1 and L2 penalties, which can result in better performance than a model with either one or the other penalty on problems.
# 
# Loss function = OLS + [ alpha l1_ratio L1-norm ] + [ 0.5 alpha (1 - l1_ratio) * L2-norm ]
# In addition to setting an alpha value, Elastic-Net also allows us to tune the l1-ratio parameter where l1-ratio = 0 corresponds to ridge and l1-ratio = 1 corresponds to lasso. We again use cross-validated Elastic-Net regression to tune hyperparameters:

# In[154]:


elastic_cv_model = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, .995, .998, 1], 
                                eps=0.001, n_alphas=100, cv=3, max_iter=100000)

# In[155]:


elastic_cv_model.fit(X_train, y_train)


# In[156]:


elastic_cv_model.l1_ratio_


# In[157]:


elastic_cv_model.alpha_


# Step 13.2: Elastic-Net Model Evaluation
# 

# In[158]:


model_evaluation(elastic_cv_model, X_test, y_test, 'Elastic-Net Reg. Test')


# In[159]:


model_evaluation(elastic_cv_model, X_train, y_train, 'Elastic-Net Reg. Train')


# The accuracy of the model on the training and test data are close to each other, so the model is not overfited.
# The accuracy of the Elastic-Net model has decreased slightly compared to Ridge model.
# 
# 
# - Step 13.3: Elastic-Net Regression Coefficients
# The ElasticNet Regression coefficients are:

# In[160]:


elastic_coef = elastic_cv_model.coef_
elastic_coef

# In[161]:


elastic_coef = pd.DataFrame(elastic_cv_model.coef_, index=X_poly.columns, columns=['ElasticNet Coef.'])
elastic_coef = elastic_coef[elastic_coef['ElasticNet Coef.']!=0]
elastic_coef.T

# Step 13.4: Elastic-Net Model Assumptions Investigation¶
# 

# In[162]:


linear_assumption(elastic_cv_model, X_test, y_test)


# In[163]:


normal_errors_assumption(elastic_cv_model, X_test, y_test)


# In[164]:


multicollinearity_assumption(X_poly[elastic_coef.index]).T


# In[165]:


autocorrelation_assumption(elastic_cv_model, X_test, y_test)


# In[166]:


homoscedasticity_assumption(elastic_cv_model, X_test, y_test)


# The assumptions are fulfilled to a good extent, but the accuracy of the model is lower than the ridge model.¶

# - Step 14: Build Higher Order Regularized Polynomial Model
# 
# Model overfitting occurs when the model learns well from train data, so it performs worst on the test data or any unseen data provided. One of the ways to avoid overfitting is to use regularization. In the overfit model, the coefficients are generally inflated. Regularization adds a penalty to the coefficients of the model and prevents them from being heavy. Therefore, when we use regularized regression models including ridge, lasso and elastic-net, it is possible to use higher order features in the model structure.

# Step 14.1: Investigating Higher Order Regularized Polynomial Models¶
# 

# The following function produces high-order features from the 2nd order to the desired order and trains and evaluates regularized models on the set of features of each order and then it returns the r2-score and the number of features used by each regularized model in the form of two separate dataframes:

# In[167]:


def poly_check(degree, X, y):
    ridge_scores = []
    lasso_scores = []
    elasticnet_scores = []
    
    ridge_features = []
    lasso_features = []
    elasticnet_features = []
    
    for d in range(2,degree+1):
        poly_features = PolynomialFeatures(degree=d, include_bias=False)
        X_poly = poly_features.fit_transform(X)
        X_poly = pd.DataFrame(X_poly, columns=poly_features.get_feature_names_out(X.columns))
        X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.3, random_state=0)

        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Ridge
        alphas = 10**np.linspace(10,-2,100)*0.5
        ridge_cv = RidgeCV(alphas = alphas, cv=3, scoring = 'neg_mean_squared_error')
        ridge_cv.fit(X_train, y_train)
        ridge_scores.append(ridge_cv.score(X_test,y_test))
        ridge_cols = ridge_cv.coef_[ridge_cv.coef_!=0].shape[0]
        ridge_features.append(ridge_cols)
        
        # Lasso
        lasso_cv = LassoCV(eps=0.01, n_alphas=100, max_iter=10000, cv=3)
        lasso_cv.fit(X_train, y_train)
        lasso_scores.append(lasso_cv.score(X_test,y_test))
        lasso_cols = lasso_cv.coef_[lasso_cv.coef_!=0].shape[0]
        lasso_features.append(lasso_cols)
        
        # Elastic-Net
        elastic_cv = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, .995, 1], eps=0.001, n_alphas=100, cv=3, max_iter=100000)
        elastic_cv.fit(X_train, y_train)
        elasticnet_scores.append(elastic_cv.score(X_test,y_test))
        elasticnet_cols = elastic_cv.coef_[elastic_cv.coef_!=0].shape[0]
        elasticnet_features.append(elasticnet_cols)
           
    scores = pd.DataFrame({'Ridge':ridge_scores, 'Lasso':lasso_scores, 'ElasticNet':elasticnet_scores}).round(4)
    scores = pd.concat([pd.DataFrame({'Degree':range(2,degree+1)}), scores], axis=1)  
    feature_num = pd.DataFrame({'Ridge':ridge_features, 'Lasso':lasso_features, 'ElasticNet':elasticnet_features})
    feature_num = pd.concat([pd.DataFrame({'Degree':range(2,degree+1)}), feature_num], axis=1)
    return scores, feature_num

# In[168]:


scores, feature_num = poly_check(6, X, y)


# In[169]:


# Plot1
fig, ax = plt.subplots(1, 2, figsize=(15,6), dpi=200, gridspec_kw={'width_ratios': [3, 1]})

sns.pointplot(x=scores['Degree'], y=scores['Ridge'], color='#ff7000', label='Ridge', ax=ax[0])
sns.pointplot(x=scores['Degree'], y=scores['Lasso'], color='#0055ff', label='Lasso', ax=ax[0])
sns.pointplot(x=scores['Degree'], y=scores['ElasticNet'], color='#23bf00', label='Elastic-Net', ax=ax[0])
ax[0].set_xlabel('Polynomial Degree', fontsize=12)
ax[0].set_ylabel('R2-Score', fontsize=12)
ax[0].legend(loc='upper left')
ax[0].grid(axis='x')
ax[0].set_ylim([0.96, 0.99])

# Annotate Points
for i,j,f in zip(scores['Degree']-2, scores['Ridge'], feature_num['Ridge']):
    ax[0].text(i, j+0.0008, str(f), ha='center', color='#ff7000', weight='bold', fontsize=15)

for i,j,f in zip(scores['Degree']-2, scores['Lasso'], feature_num['Lasso']):
    ax[0].text(i, j-0.0015, str(f), ha='center', color='#0055ff', weight='bold', fontsize=15)
    
for i,j,f in zip(scores['Degree']-2, scores['ElasticNet'], feature_num['ElasticNet']):
    ax[0].text(i, j+0.0008, str(f), ha='center', color='#23bf00', weight='bold', fontsize=15)
    
# Plot2    
table = ax[1].table(cellText=scores.values, colLabels=scores.columns, loc='center')
table.auto_set_font_size(False)
table.set_fontsize(13)
ax[1].set_xticks([])
ax[1].set_yticks([])
table.scale(1, 2)

plt.suptitle('R2-Score vs. Polynomial Degree on Test Data', fontsize=20)
plt.tight_layout()
plt.show()

# The graph on the left shows the trend of r2-score changes related to three regularized models including ridge, lasso and elastic-net along with the increase in the order of the used features.
# The values annotated on the left graph are the number of features used in each model.
# The table on the right contains r2-scores for each model.
# An optimal model is one that not only uses fewer features, but also has a high r2-score.
# In the situation where the simple polynomial model was overfit for the 3d-order model in the step 10, but the regularized models are not overfit for the features of orders higher than 2.

# Step 14.2: Build the Optimal Model¶

# Based on the previous graph, we choose the elastic-net model based on the 4th order polynomial model as the optimal model. By increasing the order from 4 onwards, the r2-score of the model is almost constant:
# 
# 

# Step 14.2.1: Create 5th-order Polynomial Features
# 

# In[170]:


poly_features = PolynomialFeatures(degree=4, include_bias=False)
X_poly = poly_features.fit_transform(X)
X_poly = pd.DataFrame(X_poly, columns=poly_features.get_feature_names_out(X.columns))
X_poly.head()

# Step 14.2.2: Split Dataset to Train & Test Sets¶
# 

# In[171]:


X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.3, random_state=0)


# In[172]:


scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# In[173]:


final_model = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, .995, 1], eps=0.001, n_alphas=100, cv=4, max_iter=100000)

# In[174]:


final_model.fit(X_train, y_train)


# In[175]:


final_model.l1_ratio_


# In[176]:


final_model.alpha_


# In[177]:


model_evaluation(final_model, X_test, y_test, 'Final Model. Test')


# In[178]:


model_evaluation(final_model, X_train, y_train, 'Final Model. Train')

