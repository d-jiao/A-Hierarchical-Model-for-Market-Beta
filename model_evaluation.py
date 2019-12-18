#!/usr/bin/env python
# coding: utf-8

# In[175]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
from tabulate import tabulate

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
get_ipython().run_line_magic('matplotlib', 'inline')


# In[130]:


n_ind = 28

dta = dict()
for ind in range(n_ind):
    df = pd.read_csv('project/result/posterior/b_ind' + str(ind + 1) + '.csv', index_col = 0)
    dta_ = np.empty((0, 2))
    for col in df.columns:
        df_ = pd.concat((df[col][:-1], df[col][1:].reset_index().drop('index', axis = 1)), axis = 1).dropna()
        dta_ = np.vstack((dta_, df_.values))
    dta[ind] = dta_

dta_ols = dict()
for ind in range(n_ind):
    df = pd.read_csv('project/result/prior/beta/beta_ind' + str(ind + 1) + '.csv', index_col = 0)
    dta_ = np.empty((0, 2))
    for col in df.columns:
        df_ = pd.concat((df[col][:-1], df[col][1:].reset_index().drop('index', axis = 1)), axis = 1).dropna()
        dta_ = np.vstack((dta_, df_.values))
    dta_ols[ind] = dta_

df_std = pd.DataFrame()
df_avg = pd.DataFrame()
for ind in range(n_ind):
    df1 = pd.read_csv('project/result/prior/std/std_ind' + str(ind + 1) + '.csv', index_col = 0)
    df2 = pd.read_csv('project/result/prior/beta/beta_ind' + str(ind + 1) + '.csv', index_col = 0)
    df_std = pd.concat((df_std, df1), axis = 1)
    df_avg = pd.concat((df_avg, df2), axis = 1)
sg = df_std.mean(axis = 1)
mu = df_avg.mean(axis = 1)

dta_vck = dict()
for ind in range(n_ind):
    df1 = pd.read_csv('project/result/prior/std/std_ind' + str(ind + 1) + '.csv', index_col = 0)
    df2 = pd.read_csv('project/result/prior/beta/beta_ind' + str(ind + 1) + '.csv', index_col = 0)
    
    sg1 = pd.DataFrame([sg for i in range(df1.shape[1])]).T
    sg1.columns = df1.columns
    mu1 = pd.DataFrame([mu for i in range(df1.shape[1])]).T
    mu1.columns = df1.columns

    b_vck = df2 * sg1 / (sg1 + df1) + mu1 * df1 / (sg1 + df1)
    b_vck.to_csv('project/result/vck/vck_ind' + str(ind + 1) + '.csv')
    
    dta_ = np.empty((0, 2))
    for col in b_vck.columns:
        df_ = pd.concat((b_vck[col][:-1], b_vck[col][1:].reset_index().drop('index', axis = 1)), axis = 1).dropna()
        dta_ = np.vstack((dta_, df_.values))
    dta_vck[ind] = dta_

r2 = np.empty((0, 3))
rmse = np.empty((0, 3))
intercept = np.empty((0, 3))
slope = np.empty((0, 3))
for ind in range(n_ind):
    X = np.empty(shape=dta[ind].shape, dtype=np.float)
    X[:, 0] = 1
    X[:, 1] = dta[ind][:, 0]
    Y = dta[ind][:, 1]

    model = sm.OLS(Y, X)
    result = model.fit()
    
    X_ols = np.empty(shape=dta_ols[ind].shape, dtype=np.float)
    X_ols[:, 0] = 1
    X_ols[:, 1] = dta_ols[ind][:, 0]
    Y_ols = dta_ols[ind][:, 1]

    model_ols = sm.OLS(Y_ols, X_ols)
    result_ols = model_ols.fit()
    
    X_vck = np.empty(shape=dta_vck[ind].shape, dtype=np.float)
    X_vck[:, 0] = 1
    X_vck[:, 1] = dta_vck[ind][:, 0]
    Y_vck = dta_vck[ind][:, 1]

    model_vck = sm.OLS(Y_vck, X_vck)
    result_vck = model_vck.fit()
    
    r2 = np.vstack((r2, [result.rsquared, result_ols.rsquared, result_vck.rsquared]))
    rmse = np.vstack((rmse, [result.mse_resid, result_ols.mse_resid, result_vck.mse_resid]))
    intercept = np.vstack((intercept, [result.params[0], result_ols.params[0], result_vck.params[0]]))
    slope = np.vstack((slope, [result.params[1], result_ols.params[1], result_vck.params[1]]))

b = pd.read_csv('project/result/posterior/b_i.csv', index_col = 0)
fig, ax = plt.subplots(figsize = (6, 8))
ax.plot(range(2005, 2019), b)
ax.legend(b.columns, loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_xlabel('Year')
ax.set_ylabel('Beta')
fig.savefig('b.png')

plt.plot(r2)
plt.legend(['Hierarchical', 'Vasicek', 'OLS'])
plt.xlabel('Industry')
plt.ylabel('R-squared')
plt.savefig('rsq.png')

plt.plot(rmse)
plt.legend(['Hierarchical', 'Vasicek', 'OLS'], loc='upper left')
plt.xlabel('Industry')
plt.ylabel('RMSE')
plt.savefig('rmse.png')

table = [[i, rmse.mean(axis = 0)[i], intercept.mean(axis = 0)[i], slope.mean(axis = 0)[i], r2.mean(axis = 0)[i]] for i in range(3)]
headers = ['Estimation', 'RMSE', 'Intercept', 'Slope', 'R-squared']
print(tabulate(table, headers, tablefmt='pipe'))

