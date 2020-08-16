# Import and set directory:
import os
os.getcwd()

# Set path:
path = 'D:/TY Workspace/3. Data Science Portfolio/1. Completed Portfolios/P5_Walmart_Weekly_Prediction/2_Development'
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
def load_file(file):
    '''load input CSVs as a dataframe'''
    return pd.read_csv(file, encoding='latin1')


def create_list_vars(df, dt_type):
    '''create a list of variables '''
    list_vars = list(df.select_dtypes(include=dt_type))
    return list_vars


def join_data(df1, df2, join_type, key=None,
              left_index=None, right_index=None):
    '''merge the dataframes by a key'''
    df_join = pd.merge(df1, df2, how=join_type, on=key,
                       left_index=False, right_index=False)
    return df_join


def drop_column(df, var_name):
    ''' drop a column on dataframe '''
    df = df.drop(var_name, axis=1)
    return df


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


def gen_unique_id(df, var_1, var_2, var_3):
    '''generate unique id for each sales transactions'''
    df['trans_id'] = df[var_1].map(str)+'_'+df[var_2].map(str)+'_'+df[var_3].map(str)
    return df['trans_id']


def eda_encode_cat_var(df, col, num_var):
    '''encode the cat. variables by mean of a num. variable by each cat'''
    cat_dict={}
    cats = df[col].cat.categories.tolist()
    for cat in cats:
        cat_dict[cat] = df[df[col] == cat][num_var].mean()
    df[col] = df[col].map(cat_dict)


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

def feature_replacement(df):
    '''impute missing values based on specific data type and a column'''
    for column in df.columns:
        if df[column].dtype.name == 'object':
            mode = df[column].mode().iloc[0]
            df[column] = df[column].fillna(mode)
        elif df[column].dtype.name == 'float64':
            mean = df[column].mean()
            df[column] = df[column].fillna(mean)
        elif df[column].dtype.name == 'datetime64[ns]':
            pseudo_date = pd.Timestamp.max
            df[column] = df[column].fillna(pseudo_date)
        else:
            md_cols = df.columns[df.columns.str.contains(pat = 'MarkDown')]
            df[md_cols] = df[md_cols].fillna(0)
            

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
    df_stat_cat = df.describe(include='O').T
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
    plt.ylabel('transactions')
    plt.show()


def plot_bar_chart(df, var_name_1):
    '''plot a bar chart'''
    plt.figure()
    var_count_1 = df[var_name_1].value_counts()
    sns.barplot(var_count_1.index,  var_count_1.values, order=var_count_1.index, alpha=0.9)
    plt.title('Frequency chart of ' + var_name_1)
    plt.ylabel('transactions')
    plt.xlabel(var_name_1)
    plt.xticks(rotation=270)
    plt.show()

    
def plot_freq_chart(x,y,df,order):
    '''plot a frequency chart'''
    plt.figure(figsize=(8,8))
    sns.countplot(x=x, hue=y, data=df, order=order)
    plt.title('Bar chart: ' + x + ' of type  labels', fontsize=20)
    plt.xticks(rotation=270, fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel(x, fontsize=12)
    plt.ylabel('transactions', fontsize=12)
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
                annot=True, annot_kws={'size':11})
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.show()


def plot_scatter(df, var_1, var_2, color, factor=None):
    '''Scatter plot of two continuous numeric features'''
    plt.figure(figsize=(8,8))
    plt.scatter(df[var_1]*factor, df[var_2], color=color)
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
if __name__ == '__main__':
    train_data = 'sales_train.csv'
    store_data = 'sales_stores.csv'
    feature_data = 'sales_features.csv'

# Load data
df_eda = load_file(train_data)
df_store = load_file(store_data)
df_feature = load_file(feature_data)

# re-order columns on df_train:
list_order = ['Store','Dept','Date','IsHoliday','Weekly_Sales']
df_eda = df_eda[list_order]

# --- 3. ETL - metadata format --- #
# left joins on feature and store data:
list_key_1 = 'Store'
df_dimension = join_data(df_feature, df_store, 'left', list_key_1)

# left joins on df_dimension and df_walmart_sales:
list_key_2 = ['Store', 'Date', 'IsHoliday']
df_eda = join_data(df_eda, df_dimension, 'left', list_key_2)
# delete keys and dataframes:
del(df_dimension, df_feature, df_store)
del(list_key_1, list_key_2)

# --- 4. Feature Engineering --- #
# convert temperature "Fahrenheit" to "Celcius":
df_eda['Temperature'] = (df_eda['Temperature'] - 32) * 5/9

# format date as follow:
date_format = '%Y-%m-%d'
df_eda['Date'] = pd.to_datetime(df_eda['Date'])

# Create a variable: dummy holiday type:
holiday_type = ['Superbowl', 'Labor_Day', 'Thanksgiving', 'Black_Friday', 'Christmas_Eves']

# adjustments based on Walmart weekly transaction date:
conditions = [(df_eda['Date']=='2010-2-12') | # 2010/02/05 + 7 days
        (df_eda['Date']=='2011-2-11') |   # 2011/02/04 + 7 days
        (df_eda['Date']=='2012-2-3'),    # 2012/02/03 + 0 days
        (df_eda['Date']=='2010-9-10') |   # 2010/09/06 + 4 days 
        (df_eda['Date']=='2011-9-9') |   # 2011/09/05 + 4 days
        (df_eda['Date']=='2012-9-7'),    # 2012/09/03 + 0 days
        (df_eda['Date']=='2010-10-15') | # 2010/10/11 + 4 days
        (df_eda['Date']=='2011-10-14') | # 2011/10/10 + 4 days
        (df_eda['Date']=='2012-10-5'),   # 2012/10/08 + 0 days
        (df_eda['Date']=='2010-11-26') | # 2010/11/26 + 0 days
        (df_eda['Date']=='2011-11-25'),  # 2011/11/25 + 0 days
        (df_eda['Date']=='2010-12-24') | # 2010/12/24 + 0 days
        (df_eda['Date']=='2011-12-23') ] # 2011/12/23 + 0 days
    
df_eda['Holiday_Type'] = np.select(conditions, holiday_type, default='non-holidays')

# check data type:
df_eda.info()

# Drop a column: Date_String
# df_eda = drop_column(df_eda, 'Date_String')

# --- 5. ETL - merging/subsetting data --- #
# define variables:
var_label = 'Weekly_Sales'
var_id_1 = 'Store'
var_id_2 = 'Dept'
var_id_3 = 'Date'
list_id_vars = [var_id_1, var_id_2, var_id_3]

vars_num_disc = create_list_vars(df_eda, 'int64') 
vars_num_cont = create_list_vars(df_eda, 'float64')

# concatenate the two lists:
vars_num = vars_num_disc + vars_num_cont

list_unwanted = {'Store','Dept'}

vars_num = [item for item in vars_num if item not in list_unwanted]

del(vars_num_disc, vars_num_cont)

################################################
# Part 3 - Exploratory Data Analysis: Insights #
################################################
# compute top 10 rows:
df_eda.head(10)

# check any duplicates
df_eda.duplicated().sum()    

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
# convert data type: store and dept as object
convert_dt_as_custom(df_eda, 'Store', 'object')
convert_dt_as_custom(df_eda, 'Dept', 'object')

# convert data type: object to category
#convert_dt_to_cat(df_eda)
# create a list: categorical variables
vars_cat = create_list_vars(df_eda, 'object')

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
df_eda[df_eda.Temperature < -14.84]

# upper bounds (UBs)
df_eda[df_eda.Weekly_Sales > 47395]

# ---5 aggregate dataframe and compute KPIs ---
# create a id column:
gen_unique_id(df_eda, 'Store', 'Dept', 'Date')

# Define the dictionary for KPIs:
kpi_dict = {'trans_id':'nunique', 'Temperature':'mean', 'CPI':'mean', 
            'Fuel_Price':'mean', 'Unemployment':'mean', 'Weekly_Sales':'mean'}

df_agg_store = eda_agg_df_var(df_eda, 'Store', kpi_dict)

df_agg_store['Sales_Per_Transactions'] = df_agg_store.Weekly_Sales/df_agg_store.trans_id

df_agg_store = df_agg_store.sort_values(by='Sales_Per_Transactions', ascending=False)

# assessment on effect of holidays:
df_agg_holiday_type = eda_agg_df_var(df_eda, 'Holiday_Type', kpi_dict)
df_agg_holiday_type

# Print a summary KPI table by Store:
df_agg_store.rename_axis('Store')

# ---6 visualize data: uni-variate ---
# plot a histogram:    
plot_hist(df_eda, 'Fuel_Price')

plot_hist(df_eda, 'Temperature')

plot_hist(df_eda, 'CPI')

plot_hist(df_eda, 'Unemployment')

# plot a bar chart:
plot_bar_chart(df_eda, 'Store')

plot_bar_chart(df_eda, 'Dept')

# plot a frequency chart:
plot_freq_chart('Store', 'Type', df_eda, None)
plot_freq_chart('Dept', 'Type', df_eda, None)

# grouped dataframe by a categorical feature:
df_grp_type = eda_grouped_df_var(df_eda, 'Type')
df_grp_store = eda_grouped_df_var(df_eda, 'Store')

# Plot pie chart(s) by categorical features:
plot_pie_chart(df_grp_store, 'Store',
               df_grp_type, 'Type')

# ---7 visualize data: bi-variate ---
# plot a box-whisker:    
plot_box(df_eda, 'Weekly_Sales', 'Type',
        'Fuel_Price', 'Type',
        'Temperature', 'Type')

# create a dataframe:
df_eda_cm = df_eda.copy()

# check data type:
df_eda_cm.info()

# convert data type: object to category
convert_dt_to_cat(df_eda_cm)

# encode categorical variables using Weekly_Sales:
for col in df_eda_cm.columns:
    if df_eda_cm[col].dtype.name == 'category':
        eda_encode_cat_var(df_eda_cm, col, 'Weekly_Sales')

# Create a list of variables:
vars_list = list(df_eda_cm.columns)

# Delete a list of unwanted variables:
unwanted_list = {'Date', 'trans_id'}

vars_list = [item for item in vars_list if item not in unwanted_list]

# plot a correlation matrix:
plot_corr_matrix(df_eda_cm, vars_list)

# plot a cross-tabulation:
plot_crosstab(df_eda, 'Store', 'Type')
plot_crosstab(df_eda, 'Dept', 'Type')

#---- Plot a scatter plot: w numerical variables ----#
plot_scatter(df_eda, 'CPI', 'Weekly_Sales', 'orange', 100)

#---- Plot a linear regression plot: w numerical variables ----#
# Compute Pearson r for combination of X and Y:
r_size_sales = compute_pearson_r(df_eda, 'Size', 'Weekly_Sales')
r_md5_sales = compute_pearson_r(df_eda, 'MarkDown5', 'Weekly_Sales')
r_unemployment_sales = compute_pearson_r(df_eda, 'Unemployment', 'Weekly_Sales')
r_md1_md4 = compute_pearson_r(df_eda, 'MarkDown1', 'MarkDown4')
r_cpi_unemployment = compute_pearson_r(df_eda, 'CPI', 'Unemployment')

# Plot a linear regression analysis:
plot_linear_reg(df_eda, 'Size', 'Weekly_Sales',
                r_size_sales, 'green', 'Transactions')
plt.show()

plot_linear_reg(df_eda, 'MarkDown5', 'Weekly_Sales',
                r_md5_sales, 'blue', 'Transactions')
plt.show()

plot_linear_reg(df_eda, 'Unemployment', 'Weekly_Sales',
                r_unemployment_sales, 'orange', 'Transactions')
plt.show()

plot_linear_reg(df_eda, 'MarkDown1', 'MarkDown4',
                r_md1_md4, 'purple', 'Transactions')
plt.show()

plot_linear_reg(df_eda, 'CPI', 'Unemployment',
                r_cpi_unemployment, 'crimson', 'Transactions')
plt.show()