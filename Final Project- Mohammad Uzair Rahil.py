Applicant: Mohammad Uzair Rahil
Dated: 07/09/2024

Project Title: Evaluation of Regression Models Using Python
Data Used: This project employs the well-known Advertising dataset, which tracks sales revenue in relation to advertisement spending across multiple channels—TV, radio, and newspapers. The dataset contains variables that represent ad spending in these media and the corresponding sales, providing a foundation for modeling relationships between marketing investments and sales performance.
Methodology:
1.	Data Exploration and Preprocessing:
o	First, exploratory data analysis (EDA) was conducted using correlation matrices and heatmaps to understand the relationships among variables and detect multicollinearity.
o	The dataset was then preprocessed, and missing values (if any) were handled to ensure data quality for modeling.
2.	Regression Models:
o	Several linear regression models were developed, labeled lm2 through lm6. These models vary in complexity and feature combinations to assess different predictor relationships with sales.
o	Using the Scikit-learn library, the dataset was split into training and test sets to prevent overfitting and to assess the generalizability of the models.
3.	Model Evaluation:
o	The performance of each regression model was assessed using key statistical parameters:
	Coefficient of Determination (R²): Measures the proportion of variance in sales explained by the model, indicating its goodness of fit.
	Root Mean Squared Error (RMSE): Quantifies the model's prediction accuracy by calculating the square root of the average squared differences between observed and predicted values.
o	Residuals were analyzed to check for any bias in predictions, and diagnostic plots were generated to evaluate the model's performance visually.
4.	Additional Statistical Tests:
o	To further evaluate the models, bias analysis was conducted, checking for systematic errors in predictions.
o	Residual analysis, including the distribution of residuals and residual plots, was used to confirm the assumptions of linearity and homoscedasticity (constant variance of errors).
Results:
The lm6 model, which incorporated a specific combination of predictor variables and interactions, demonstrated superior performance compared to other models (lm2 through lm5). It achieved the highest R² value, indicating a stronger fit to the data, and the lowest RMSE, reflecting the most accurate predictions. Residual plots also indicated minimal bias, affirming the model's robustness in predicting sales based on advertisement spending.



code begins here:


#!/usr/bin/env python
# coding: utf-8

# # Final Project- Mohammad Uzair Rahil

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import skew
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


import matplotlib.pyplot as plt
plt.style.use("ggplot")
plt.rcParams['figure.figsize'] = (12, 8)


# In[5]:


advert = pd.read_csv('Advertising.csv')
advert.head()


# In[6]:


advert.info()


# In[7]:


sns.pairplot(advert, x_vars=['TV','radio','newspaper'], y_vars='sales', height=7, aspect=0.7);


# In[8]:


from sklearn.linear_model import LinearRegression

# create X and y
feature_cols = ['TV', 'radio', 'newspaper']
X = advert[feature_cols]
y = advert.sales

# instantiate and fit
lm1 = LinearRegression()
lm1.fit(X, y)

# print the coefficients
print(lm1.intercept_)
print(lm1.coef_)


# In[9]:


# pair the feature names with the coefficients
list(zip(feature_cols, lm1.coef_))


# In[10]:


sns.heatmap(advert.corr(), annot=True)


# In[11]:


from sklearn.metrics import r2_score

lm2 = LinearRegression().fit(X[['TV', 'radio']], y)
lm2_preds = lm2.predict(X[['TV', 'radio']])

print("R^2: ", r2_score(y, lm2_preds))


# In[12]:


lm3 = LinearRegression().fit(X[['TV', 'radio', 'newspaper']], y)
lm3_preds = lm3.predict(X[['TV', 'radio', 'newspaper']])

print("R^2: ", r2_score(y, lm3_preds))


# In[13]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X = advert[['TV', 'radio', 'newspaper']]
y = advert.sales

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1)

lm4 = LinearRegression()
lm4.fit(X_train, y_train)
lm4_preds = lm4.predict(X_test)

print("RMSE :", np.sqrt(mean_squared_error(y_test, lm4_preds)))
print("R^2: ", r2_score(y_test, lm4_preds))


# In[14]:


X = advert[['TV', 'radio']]
y = advert.sales

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1)

lm5 = LinearRegression()
lm5.fit(X_train, y_train)
lm5_preds = lm5.predict(X_test)

print("RMSE :", np.sqrt(mean_squared_error(y_test, lm5_preds)))
print("R^2: ", r2_score(y_test, lm5_preds))


# In[21]:


from yellowbrick.regressor import PredictionError, ResidualsPlot
visualizer = PredictionError(lm5)

visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
visualizer.poof() 


# In[22]:


pip install yellowbrick


# In[23]:


visualizer = ResidualsPlot(lm5)
visualizer.fit(X_train, y_train)  
visualizer.score(X_test, y_test) 
visualizer.poof()


# In[24]:


advert['interaction'] = advert['TV'] * advert['radio']


# In[25]:


X = advert[['TV', 'radio', 'interaction']]
y = advert.sales

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1)

lm6 = LinearRegression()
lm6.fit(X_train, y_train)
lm6_preds = lm6.predict(X_test)

print("RMSE :", np.sqrt(mean_squared_error(y_test, lm6_preds)))
print("R^2: ", r2_score(y_test, lm6_preds))


# In[26]:


visualizer = PredictionError(lm6)

visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
visualizer.poof() 


# In[ ]:




