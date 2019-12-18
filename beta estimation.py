import numpy as np
import pandas as pd
import statsmodels.api as sm
import h5py
import os
os.chdir('/Users/jiayihe/desktop/Columbia MFE/5224')
import scipy.io as scio
import math

filename1 = 'project/closeprice.mat'
data_closeprice = scio.loadmat(filename1)
filename2 = 'project/sector.mat'
data_sector = scio.loadmat(filename2)
filename3 = 'project/date.mat'
data_date = scio.loadmat(filename3)



##---------------------Data Processing--------------------##

#get different sectors
sector = data_sector['sector'][0][0]
col_name = []
for i in range(28):
    name = 'ind' + str(i+1)
    col_name.append(name)
ind_dic = {}
ind_dic['ind1'] = list(set(list(sector[0][0])))
sector2 = sector[1][0]
for i in range(1,28):
    ind_dic[col_name[i]] = list(set(list(sector2[i-1][0])))

#get individuaul stock return table
closeprice = data_closeprice['closeprice']
closeprice = closeprice.T
price_df = pd.DataFrame(closeprice)
date = data_date['tdate']
price = np.hstack((date,closeprice))
close_df = pd.DataFrame(price)
ret = np.log(close_df/close_df.shift(1))
ret.drop([0],axis =1,inplace =True)
ret['date'] = date
ret.drop([0],inplace = True)

#get benchmark return table
mkt_ret = pd.read_csv('project/szzz.csv')
mkt_ret = mkt_ret[['Date','Ret']]
mkt_ret.columns = ['date','Ret']
mkt_ret['date'] = pd.to_datetime(mkt_ret['date'])
f = lambda x: int(x.year * 1e4 + x.month * 1e2 + x.day)
mkt_ret['date'] = mkt_ret['date'].apply(f)
df_ret = pd.merge(ret,mkt_ret,how='left',on = 'date')
df_ret['Ret'] = df_ret['Ret'].fillna(0)


##---------------------Calculate Prior--------------------##

def find_dates(year, date_full):
    ind = np.logical_and((date_full - year * 10e3) > 0, (date_full - year * 10e3) < 10e3)
    return ind
indicator = find_dates(2005,df_ret.date)
year_ret = df_ret[indicator]
test = col_name[0]
ind_li = ind_dic[test]
sec1 = year_ret[ind_li]
mkt1 = year_ret['Ret']


#get beta ij,sigma ij
years = range(2005, 2019)
for sec in col_name:
    beta_dic = {}
    filename_beta = 'project/result/prior/beta/beta_'+sec+'.csv'
    std_dic = {}
    filename_std = 'project/result/prior/std/std_'+sec+'.csv'
    for year in years:
        indicator = find_dates(year,df_ret.date)
        year_ret = df_ret[indicator]
        ind_li = ind_dic[sec]
        sec1 = year_ret[ind_li]
        mkt1 = year_ret['Ret']
        for company in ind_dic[sec]:
            print('doing', year,company)
            Y = sec1[company]
            X = mkt1
            X = sm.add_constant(X)
            if Y.isna().sum()>20 or (Y==0).sum()>20:
                beta = np.nan
                std = np.nan
            else:
                try:
                    model = sm.OLS(Y,X)
                    results = model.fit()
                    beta = results.params[1]
                    std = results.bse[1]
                except:
                    beta = np.nan
                    std = np.nan
            if company not in beta_dic:
                beta_dic[company] = [beta]
                std_dic[company] = [std]
            else:
                beta_dic[company].append(beta)
                std_dic[company].append(std)
    beta_table = pd.DataFrame(beta_dic)
    std_table = pd.DataFrame(std_dic)
    beta_table.to_csv(filename_beta)
    std_table.to_csv(filename_std)

#get sigma i
file_path = 'project/result/prior'
sigma_i = {}
for sec in col_name:
    df_std = pd.read_csv(file_path+'/std/std_'+sec+'.csv')
    np_std = np.array(df_std)
    for year in range(14):
        np_std_year = np_std[year][1:]
        num_sec = np_std_year.shape[0]-np.isnan(np_std_year).sum()
        where_are_nan = np.isnan(np_std_year)
        np_std_year[where_are_nan] = 0
        sigma = np.power(np_std_year,2).sum()/num_sec

        if sec not in sigma_i:
            sigma_i[sec] = [sigma]
        else:
            sigma_i[sec].append(sigma)
sigma_i_df = pd.DataFrame(sigma_i)

#get sigma
sec_num = 28
np_sigma_i = np.array(sigma_i_df)
sigma = []

for year in range(14):
    sigma_i_year= np_sigma_i[year][1:]
    sigma_year = sigma_i_year.sum()/sec_num
    sigma.append(sigma_year)

##---------------------Calculate Posterior--------------------##
#get sector porsterior
sigma_path = 'project/result/prior'
mu_i = {}
for sec in col_name:
    df_beta = pd.read_csv(file_path + '/beta/beta_' + sec + '.csv', index_col=0)
    for year in range(14):

        df_beta_year = df_beta.iloc[year]
        sum_beta = df_beta_year.sum()
        num_beta = df_beta_year.shape[0] - df_beta_year.isna().sum()

        sigma_year = sigma[year]
        sigma_i_year = sigma_i_df[sec][year]

        mu = (1 / sigma_year + sum_beta / sigma_i_year) / (1 / sigma_year + num_beta / sigma_i_year)

        if sec not in mu_i:
            mu_i[sec] = [mu]
        else:
            mu_i[sec].append(mu)
mu_i_df = pd.DataFrame(mu_i)

sigma_path = 'project/result/prior'
sigma_i_hat = {}
for sec in col_name:
    df_beta = pd.read_csv(file_path + '/beta/beta_' + sec + '.csv', index_col=0)
    for year in range(14):

        df_beta_year = df_beta.iloc[year]
        # sum_beta = df_beta_year.sum()
        num_beta = df_beta_year.shape[0] - df_beta_year.isna().sum()

        sigma_year = sigma[year]
        sigma_i_year = sigma_i_df[sec][year]

        sigma_hat = 1 / (1 / sigma_year + num_beta / sigma_i_year)

        if sec not in sigma_i_hat:
            sigma_i_hat[sec] = [sigma_hat]
        else:
            sigma_i_hat[sec].append(sigma_hat)
sigma_i_hat_df = pd.DataFrame(sigma_i_hat)


#get single stock posterior
sigma_path = 'project/result/prior/std/'
beta_path = 'project/result/prior/beta/'
posterior_path_beta = 'project/result/posterior/b_ij/'
posterior_path_sigma = 'project/result/posterior/sigma_ij/'

years = range(14)
for sec in col_name:
    b_dic = {}
    sigma_dic = {}
    filename_b = posterior_path_beta+'b_'+sec+'.csv'
    filename_sigma = posterior_path_sigma+'sigma_'+sec+'.csv'
    for year in years:
        mu_i = mu_i_df[sec][year]
        sigma_i = sigma_i_hat_df[sec][year]
        beta_df = pd.read_csv(beta_path+'beta_'+sec+'.csv',index_col = 0)
        sigma_df = pd.read_csv(sigma_path+'std_'+sec+'.csv',index_col = 0)
        for company in ind_dic[sec]:
            print('doing', year,company)
            beta_ij = beta_df[str(company)][year]
            std_ij = sigma_df[str(company)][year]
            try:
                b_ij = (mu_i/sigma_i+beta_ij/std_ij)/(1/sigma_i+1/std_ij)
            except:
                b_ij = np.nan
            try:
                sigma_ij = 1/(1/sigma_i+1/std_ij)
            except:
                sigma_ij = np.nan
            if company not in b_dic:
                b_dic[company] = [b_ij]
                sigma_dic[company] = [sigma_ij]
            else:
                b_dic[company].append(b_ij)
                sigma_dic[company].append(sigma_ij)
    b_table = pd.DataFrame(b_dic)
    sigma_table = pd.DataFrame(sigma_dic)
    b_table.to_csv(filename_b)
    sigma_table.to_csv(filename_sigma)