# Import and set directory:
import os
os.getcwd()

# Set path:
path = 'D:/TY Workspace/3. Data Science Portfolio/1. Completed Portfolios/P2_Insurance_Propensity_Prediction/2_Development'
os.chdir(path)

# Import all libraries #
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import csv
from sklearn.utils import shuffle
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

# Authorization #
__author__ = "Taesun Yoo"
__email__ = "yoots1988@gmail.com"

#################################
# Part 2 - Discover the Problem #
#################################
# --- 1. Write Out List of Functions --- #
# --- 1. Write Out List of Functions --- #
def load_file(file):
    '''load input CSVs as a dataframe'''
    return pd.read_csv(file, encoding='latin1')


def convert_dt_as_date(df, var_name, date_format):
    '''convert the variable as specified date format'''
    df[var_name] = pd.to_datetime(df[var_name], format=date_format)
    return df[var_name]


def convert_dt_as_custom(df, var_name, dt_type):
    '''convert datatype on selected variables'''
    df[var_name] = df[var_name].astype(dt_type)
    return df[var_name]


def convert_dt_to_cat(df):
    '''convert data type to category'''
    for col in df.columns:
        if df[col].dtype.name == 'object':
            df[col] = df[col].astype('category')


def eda_encode_cat_var(df, col, num_var):
    '''encode the cat. variables by mean of a num. variable by each cat'''
    cat_dict={}
    cats = df[col].cat.categories.tolist()
    for cat in cats:
        cat_dict[cat] = df[df[col] == cat][num_var].mean()
    df[col] = df[col].map(cat_dict)


#def pandas_qcut(df, new_var, old_var, q):
#    ''' apply pd.qcut for equal size binning '''
#    df[new_var] = pd.qcut(df[old_var], q=q, duplicates='drop',
#                          precision=0, labels=None)
#    return df[new_var]


def estimate_width(df, var_name, interval_size):
    ''' estimate the width of continuous variable '''
    width = (max(df[var_name]) - min(df[var_name]))/interval_size
    df_estimate = pd.DataFrame([[min(df[var_name])-1, max(df[var_name])+1, round(width)]])
    df_estimate.columns = ['min', 'max', 'width']
    return df_estimate


def pandas_cut(df, new_var, old_var, bins, labels=None):
    ''' apply pd.cut for custom size binning '''
    df[new_var] = pd.cut(df[old_var], bins=bins, 
                          labels=labels, right=False)
    return df[new_var]


def check_value_counts(df, var_name):
    '''return grouped value counts'''
    grouped_counts = df[var_name].value_counts()
    return grouped_counts


def feature_replacement(X):
    ''' replace missing values based on specific data type of a column '''
    for col in X.columns:
        if X[col].dtype.name == 'object':
            mode = X[col].mode().iloc[0]
            X[col] = X[col].fillna(mode)
        elif X[col].dtype.name == 'float64':
            mean = X[col].mean()
            X[col] = X[col].fillna(mean)
        else:
            X[col].dtype.name == 'int64'
            median = X[col].median()
            X[col] = X[col].fillna(median)
            

def eda_missing(df):
    '''compute missing % on each var'''
    df_missing = pd.DataFrame(df.isnull().sum(), columns=['count'])
    df_missing['pct'] = (df_missing['count']/len(df)) * 100
    return df_missing


def eda_stat_num(df):
    ''' perform eda for numerical features '''
    df_stat_num = df.describe().T
    df_stat_num = df_stat_num[['count', 'min', 'mean', 'max', 'std', '25%', '50%', '75%']]
    df_stat_num = pd.DataFrame(df_stat_num)
    return df_stat_num


def eda_stat_cat(df):
    ''' perform eda for categorical features '''
    df_stat_cat = df.describe(include='category').T
    df_stat_cat = pd.DataFrame(df_stat_cat)
    return df_stat_cat


def eda_outliers(df):
    '''check outliers using the IQR method'''
    df['IQR'] = df['75%'] - df['25%']
    df['LB']  = df['25%'] - 1.5*df['IQR']
    df['UB']  = df['75%'] + 1.5*df['IQR']
    df = df.drop(['count','std','mean','25%','50%','75%','IQR'], axis=1)
    return df


def eda_agg_df_var(df, cat_var, kpi_dict):
    '''compute aggregated dataframe to calculate the KPIs'''
    df_agg = df.groupby(by=cat_var).agg(kpi_dict)
    return df_agg


def eda_grouped_df_var(df, cat_var):
    '''create a grouped dataframe by categorical variable'''
    df_grp = pd.DataFrame(df.groupby([cat_var])[cat_var].count())
    df_grp.columns = ['count']
    return df_grp


def plot_hist(df, var_1):
    '''plot a histogram'''
    plt.figure()
    print("skenewss is:", df[var_1].skew())
    df[var_1].hist(color='green')
    plt.title('Histogram of ' + var_1)
    plt.xlabel(var_1)
    plt.ylabel('patients')
    plt.show()


def plot_bar_chart(df, var_name_1):
    '''plot a bar chart'''
    plt.figure()
    var_count_1 = df[var_name_1].value_counts()
    sns.barplot(var_count_1.index,  var_count_1.values, alpha=0.9)
    plt.title('Frequency chart of ' + var_name_1)
    plt.ylabel('patients')
    plt.xlabel(var_name_1)
    plt.show()

    
def plot_freq_chart(x,y,df,order):
    '''plot a frequency chart'''
    plt.figure(figsize=(8,8))
    sns.countplot(x=x, hue=y, data=df, order=order)
    plt.title('Bar chart: ' + x + ' of patient group labels', fontsize=20)
    plt.xticks(rotation=270, fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel(x, fontsize=12)
    plt.ylabel('patients', fontsize=12)
    plt.legend(loc='upper right', fontsize=20)
    plt.show()


def plot_pie_chart(df_1, var_name_1,
                   df_2, var_name_2):
    '''plot a pie chart of specified variables'''
    plt.figure(figsize=(15,15))
    # Sub-plot 1:
    plt.subplot(1,2,1)
    plt.pie(df_1, autopct='%.0f%%', wedgeprops={'edgecolor':'white'},
            textprops={'fontsize':15})
    plt.title('Pie Chart of '+ var_name_1)
    plt.legend(labels = df_1.index, loc='upper right')
    # Sub-plot 2:
    plt.subplot(1,2,2)
    plt.pie(df_2, autopct='%.0f%%', wedgeprops={'edgecolor':'white'},
            textprops={'fontsize':15})
    plt.title('Pie Chart of '+ var_name_2)
    plt.legend(labels = df_2.index, loc='upper right')
    plt.show()
    

def plot_box(df, num_var_1, cat_var_1,
             num_var_2, cat_var_2, 
             num_var_3, cat_var_3, hue=None):
    '''plot a box-whisker of specified variables'''
    plt.figure(figsize=(15,15))
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    # Sub-plot 1:
    plt.subplot(1,3,1)
    df.sort_values(by=[num_var_1], inplace=True)
    sns.set(style='whitegrid')
    sns.boxplot(cat_var_1, num_var_1, hue, df)
    plt.title('Box plot of ' + num_var_1 + ' by ' + cat_var_1)
    plt.xticks(rotation=270, fontsize=10)
    # Sub-plot 2:
    plt.subplot(1,3,2)
    df.sort_values(by=[num_var_2], inplace=True)
    sns.set(style='whitegrid')
    sns.boxplot(cat_var_2, num_var_2, hue, df)
    plt.title('Box plot of ' + num_var_2 + ' by ' + cat_var_2)
    plt.xticks(rotation=270, fontsize=10)
    # Sub-plot 3:
    plt.subplot(1,3,3)
    df.sort_values(by=[num_var_3], inplace=True)
    sns.set(style='whitegrid')
    sns.boxplot(cat_var_3, num_var_3, hue, df)
    plt.title('Box plot of ' + num_var_3 + ' by ' + cat_var_3)
    plt.xticks(rotation=270, fontsize=10)    

    
def plot_crosstab(df, cat_var_1, cat_var_2):
    '''plot a crosstab of two categorical variables'''
    table = pd.crosstab(df[cat_var_1], df[cat_var_2])
    return table


def plot_corr_matrix(df, list_vars):
    ''' plot a correlation matrix '''
    corr = df[list_vars].corr()
    # Create a mask
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(corr)] = True
    plt.figure(figsize=(12,12))
    sns.heatmap(corr, mask=mask, square=True, linewidths = .5,
                cmap=sns.diverging_palette(220,10,as_cmap=True),
                vmin = -1, vmax = 1, fmt=".2f",
                annot=True, annot_kws={'size':15})
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.show()


def plot_scatter(df, var_1, var_2, color, factor=None):
    '''Scatter plot of two continuous numeric features'''
    plt.figure(figsize=(8,8))
    plt.scatter(df[var_1], df[var_2], color=color)
    plt.title('Relationship between '+ var_1 + ' and ' + var_2)
    plt.xlabel(var_1)
    plt.ylabel(var_2)
    plt.show()

    
def compute_pearson_r(df, var_x, var_y):
    '''compute Pearson r correlation'''
    corr_mat = np.corrcoef(df[var_x],df[var_y])
    return corr_mat[0, 1]


def plot_linear_reg(df, var_x, var_y, 
                    pearson_r, color, label):
    '''plot a pair of linear regressions'''
    plt.figure(figsize=(10,10))
    plt.plot(df[var_x], df[var_y], 'r--', label='pearson_r =%.2f' % pearson_r,
             marker='.', linestyle='none', color=color)
    plt.margins(0.02)
    plt.legend(loc='upper left')
    plt.xlabel(var_x)
    plt.ylabel(var_y)
    plt.title(var_x + ' vs. ' + var_y + ' by ' + label)
    # Fit linear regression:
    a,b = np.polyfit(df[var_x], df[var_y], 1)
    x = np.array([min(df[var_x]), max(df[var_x])])
    y = a*x + b
    plt.plot(x,y)


# --- 2. Load the data --- #
# define input CSVs:
if __name__ == '__main__':
    eda_file = 'insurance_train.csv'

# load data:
df_eda = load_file(eda_file)

# map column names to lowercase:
df_eda.columns = map(str.lower, df_eda.columns)

# check data types:
df_eda.info()

# feature engineering:
# create age from age_in_days
df_eda['age'] = df_eda['age_in_days']/365.25
df_eda['age'] = df_eda['age'].astype('int64')

# drop age_in_days:
df_eda.drop('age_in_days', axis=1, inplace=True) 

# define variables:
vars_cat = list(df_eda.select_dtypes(include='object').columns)
vars_num_disc = list(df_eda.select_dtypes(include='int64'))
vars_num_cont = list(df_eda.select_dtypes(include='float64'))

# concatenate a list:
vars_num = vars_num_disc + vars_num_cont

# delete a list of unwanted variables:
unwanted_list = {'id', 'renewal'}
vars_num = [var for var in vars_num if var not in unwanted_list]

var_label = 'renewal'
var_id = 'id'

################################################
# Part 3 - Exploratory Data Analysis: Insights #
################################################
# compute top 10 rows:
df_eda.head(10)
    
# create a missing % dataframe: before
df_missing_pre = eda_missing(df_eda)
df_missing_pre

# feature replacement:
feature_replacement(df_eda)

# create a missing % dataframe: after
df_missing_post = eda_missing(df_eda)
df_missing_post

del(df_missing_pre, df_missing_post)

# ---3 perform summary statistics ---
# convert data type: object to category
convert_dt_to_cat(df_eda)

# numerical features:
df_stat_num = eda_stat_num(df_eda[vars_num])
df_stat_num

# categorical features:
df_stat_cat = eda_stat_cat(df_eda[vars_cat])
df_stat_cat

# ---4 detect outliers ---
# create a dataframe for IQR:
df_outliers = eda_outliers(df_stat_num)
df_outliers

# check outliers:
# lower bounds (LBs)
df_eda[df_eda.application_underwriting_score < 97.795]

# upper bounds (UBs)
df_eda[df_eda.income > 468210]

# define bin_size and labels:
# age_group:
estimate_width(df_eda, 'age', 4)
bins_age = [19, 40, 60, 80, 103]
labels_age = ['Age 20-40s', 'Age 40-60s', 'Age 60-80s', 'Age +80']

# premium_group:
estimate_width(df_eda, 'premium', 4)
bins_premium = [1199, 15900, 30600, 45300, 60001]

# income_group: extremely right skewed
estimate_width(df_eda, 'income', 4)
bins_income = [24000, 69000, 114000, 159000, 
               204000, 294000, 339000, 384000,
               429000, 474000, 90262601]

# num_prem_paid_group:
estimate_width(df_eda, 'no_of_premiums_paid', 10)
bins_prem_paid = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 61]

# cnt_3_6_mo_late_group:
estimate_width(df_eda, 'count_3-6_months_late', 4)
bins_cnt_3_6_mo = [0, 3, 6, 9, 12, 15]

# cnt_6_12_mo_late_group:
estimate_width(df_eda, 'count_6-12_months_late', 4)
bins_cnt_6_12_mo = [0, 4, 8, 12, 16, 20]

# cnt_over_12_mo_late_group:
estimate_width(df_eda, 'count_more_than_12_months_late', 4)
bins_cnt_over_12_mo = [0, 3, 6, 9, 12, 15]

# feature discretization:
df_eda['age_group'] = pandas_cut(df_eda, 'age_group', 'age', bins=bins_age, labels=labels_age)
df_eda['premium_group'] = pandas_cut(df_eda, 'premium_group', 'premium', bins=bins_premium)
df_eda['income_group'] = pandas_cut(df_eda, 'income_group', 'income', bins=bins_income)
df_eda['num_prem_paid_group'] = pandas_cut(df_eda, 'num_prem_paid_group', 'no_of_premiums_paid', bins=bins_prem_paid)
df_eda['cnt_3_6_mo_late_group'] = pandas_cut(df_eda, 'cnt_3_6_mo_late_group', 'count_3-6_months_late', bins=bins_cnt_3_6_mo)
df_eda['cnt_6_12_mo_late_group'] = pandas_cut(df_eda, 'cnt_6_12_mo_late_group', 'count_6-12_months_late', bins=bins_cnt_6_12_mo)
df_eda['cnt_over_12_mo_late_group'] = pandas_cut(df_eda, 'cnt_over_12_mo_late_group', 'count_more_than_12_months_late', bins=bins_cnt_over_12_mo)

# check engineered categorical groups:
check_value_counts(df_eda, 'age_group')
check_value_counts(df_eda, 'premium_group')
check_value_counts(df_eda, 'income_group')
check_value_counts(df_eda, 'num_prem_paid_group')
check_value_counts(df_eda, 'cnt_3_6_mo_late_group')
check_value_counts(df_eda, 'cnt_6_12_mo_late_group')
check_value_counts(df_eda, 'cnt_over_12_mo_late_group')


# ---5 aggregate dataframe and compute KPIs ---
# Define the dictionary for KPIs:
kpi_dict = {'id':'nunique', 'age':'mean', 'income':'mean', 
            'premium':'mean', 'no_of_premiums_paid': 'mean',
            'count_3-6_months_late':'mean', 'count_6-12_months_late':'mean',
            'perc_premium_paid_by_cash_credit':'mean', 'application_underwriting_score':'mean'}

df_agg_renewal = eda_agg_df_var(df_eda, 'renewal', kpi_dict)

# Print a summary KPI table by Worker Region:
df_agg_renewal

# ---6 visualize data: uni-variate ---
# plot a histogram:    
plot_hist(df_eda, 'age')

plot_hist(df_eda, 'premium')

plot_hist(df_eda, 'perc_premium_paid_by_cash_credit')

plot_hist(df_eda, 'application_underwriting_score')

vars_cat

# plot a bar chart:
plot_bar_chart(df_eda, 'sourcing_channel')

plot_bar_chart(df_eda, 'residence_area_type')

# plot a frequency chart:
plot_freq_chart('age_group', 'renewal', df_eda, None)

plot_freq_chart('income_group', 'renewal', df_eda, None)

plot_freq_chart('premium_group', 'renewal', df_eda, None)

plot_freq_chart('num_prem_paid_group', 'renewal', df_eda, None)

plot_freq_chart('cnt_3_6_mo_late_group', 'renewal', df_eda, None)

# grouped dataframe by a categorical feature:
df_grp_renewal = eda_grouped_df_var(df_eda, 'renewal')
df_grp_residence_type = eda_grouped_df_var(df_eda, 'residence_area_type')
df_grp_age_group = eda_grouped_df_var(df_eda, 'age_group')
df_grp_sourcing_channel = eda_grouped_df_var(df_eda, 'sourcing_channel')

# Plot pie chart(s) by categorical features:
plot_pie_chart(df_grp_renewal, 'Renewal',
               df_grp_residence_type, 'Residence Type')

plot_pie_chart(df_grp_age_group, 'Age Group',
               df_grp_sourcing_channel, 'Sourcing Channel')

# ---7 visualize data: bi-variate ---
# plot a box-whisker:    
plot_box(df_eda, 'age', 'renewal',
        'premium', 'renewal',
        'perc_premium_paid_by_cash_credit', 'renewal')

# create a dataframe:
df_eda_cm = df_eda.copy()

# check data type:
df_eda_cm.info()

# convert data type: object to category
convert_dt_to_cat(df_eda_cm)

# encode categorical variables using age:
for col in df_eda_cm.columns:
    if df_eda_cm[col].dtype.name == 'category':
        eda_encode_cat_var(df_eda_cm, col, 'age')

# Create a list of variables:
vars_list = list(df_eda_cm.columns)

# Delete a list of unwanted variables:
unwanted_list = {'sourcing_channel', 'residence_area_type', 
                 'age_group', 'premium_group', 'num_prem_paid_group',
                 'cnt_3_6_mo_late_group', 'cnt_6_12_mo_late_group', 
                 'cnt_over_12_mo_late_group', 'id'}

vars_list = [item for item in vars_list if item not in unwanted_list]

# plot a correlation matrix:
plot_corr_matrix(df_eda_cm, vars_list)

# plot a cross-tabulation:
plot_crosstab(df_eda, 'age_group', 'renewal')

plot_crosstab(df_eda, 'sourcing_channel', 'renewal')

#---- Plot a linear regression plot: w numerical variables ----#
# Compute Pearson r for age and no_of_premiums_paid:
r_age_num_prem_paid= compute_pearson_r(df_eda, 'age', 'no_of_premiums_paid')
print(r_age_num_prem_paid) 

r_age_pct_paid_cash_credit = compute_pearson_r(df_eda, 'age', 'perc_premium_paid_by_cash_credit')
print(r_age_pct_paid_cash_credit) 

r_income_premium = compute_pearson_r(df_eda, 'income', 'premium')
print(r_income_premium) 

r_pct_paid_cash_credit_3_6_mo_late = compute_pearson_r(df_eda, 'perc_premium_paid_by_cash_credit', 'count_3-6_months_late')
print(r_pct_paid_cash_credit_3_6_mo_late) 

r_prem_paid_app_score = compute_pearson_r(df_eda, 'no_of_premiums_paid', 'application_underwriting_score')
print(r_prem_paid_app_score) 

# Plot a linear regression analysis:
plot_linear_reg(df_eda, 'age', 'no_of_premiums_paid',
                r_age_num_prem_paid, 'green', 'clients')
plt.show()

# Plot a linear regression analysis:
plot_linear_reg(df_eda, 'age', 'perc_premium_paid_by_cash_credit',
                r_age_pct_paid_cash_credit, 'blue', 'clients')
plt.show()

# Plot a linear regression analysis:
plot_linear_reg(df_eda, 'income', 'premium',
                r_income_premium, 'purple', 'clients')
plt.show()

# Plot a linear regression analysis:
plot_linear_reg(df_eda, 'perc_premium_paid_by_cash_credit', 'count_3-6_months_late',
                r_pct_paid_cash_credit_3_6_mo_late, 'red', 'clients')
plt.show()

# Plot a linear regression analysis:
plot_linear_reg(df_eda, 'no_of_premiums_paid', 'application_underwriting_score',
                r_pct_paid_cash_credit_3_6_mo_late, 'orange', 'clients')
plt.show()