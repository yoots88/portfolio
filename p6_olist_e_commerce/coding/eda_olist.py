# Import and set directory:
import os
os.getcwd()

# Set path:
path = 'D:/TY Workspace/3. Data Science Portfolio/1. Completed Portfolios/P6_Customer_Segments_Oilst/2_Development/Input'
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


def clean_data(df):
    clean_df = df.drop_duplicates()
    # clean_df = clean_df[clean_df['profit_margin'] > 0]
    return clean_df


def join_data(df1, df2, join_type, key=None, 
              left_index=None, right_index=None):
    '''merge the dataframes by a key'''
    df_join = pd.merge(df1, df2, how=join_type, on=key,
                       left_index=False, right_index=False)
    return df_join


def feature_replacement(df):
    ''' replace missing values based on data types of columns '''
    for col in df.columns:
        if df[col].dtype.name == 'category':
            mode = df[col].mode().iloc[0]
            df[col] = df[col].fillna(mode)
        elif df[col].dtype.name == 'datetime64[ns]':
            pseudo_date = pd.Timestamp.max
            df[col] = df[col].fillna(pseudo_date)
        elif df[col].dtype.name == 'float64':
            mean = df[col].mean()
            df[col] = df[col].fillna(mean)
        else:
            df[col].dtype.name == 'int64'
            median = df[col].median()
            df[col] = df[col].fillna(median)
            

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
    plt.ylabel('transactions')
    plt.show()


def plot_bar_chart(df, var_name_1):
    '''plot a bar chart'''
    plt.figure()
    var_count_1 = df[var_name_1].value_counts().sort_values(ascending=False)
    sns.barplot(var_count_1.index,  var_count_1.values, alpha=0.9)
    plt.title('Frequency chart of ' + var_name_1)
    plt.ylabel('transactions')
    plt.xlabel(var_name_1)
    plt.xticks(rotation=270)
    plt.show()
    
    
def plot_freq_chart(x,y,df,order):
    '''plot a frequency chart'''
    plt.figure(figsize=(8,8))
    sns.countplot(x=x, hue=y, data=df, order=order)
    plt.title('Bar chart: ' + x + ' of customer group labels', fontsize=20)
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


def write_as_csv(df, filename, filepath):
    ''' save the dataframe as .csv file in specific filepath '''
    df.to_csv(os.path.join(filepath, filename), index=False, index_label=None)

# --- 2. Load the data --- #
# define input CSVs:
if __name__ == '__main__':
    customers = 'olist_customers.csv'
    orders = 'olist_orders.csv'
    order_items = 'olist_order_items.csv'
    order_payment = 'olist_order_payments.csv'
    products = 'olist_products.csv'
    prod_name_translator = 'olist_products_name_translator.csv'
    sellers = 'olist_sellers.csv'
    # order_reviews = 'olist_order_reviews.csv'
    # geolocation = 'olist_geolocation.csv'
    
# Load data
customers = load_file(customers)
orders = load_file(orders)
order_items = load_file(order_items)
order_payment = load_file(order_payment)
products = load_file(products)
prod_name_translator = load_file(prod_name_translator)
sellers = load_file(sellers)

# rename the columns
orders.rename(columns={'order_purchase_timestamp':'order_purchase_date',
                       'order_approved_at':'order_approved_date'}, inplace=True)    

products.rename(columns={'product_category_name':'prod_cat_name', 
                         'product_name_lenght':'prod_name_length', 
                         'product_description_lenght':'prod_desc_length',
                         'product_photos_qty':'prod_photo_qty', 
                         'product_weight_g':'prod_weight_g',
                         'product_length_cm':'prod_length_cm', 
                         'product_height_cm':'prod_height_cm',
                         'product_width_cm':'prod_width_cm'}, inplace=True)    

prod_name_translator.rename(columns={'ï»¿product_category_name':'prod_cat_name',
                                     'product_category_name_english':'prod_cat_name_eng'},inplace=True)

# --- 3. ETL data transforms ---
# Join operations between data tables #
# left joins order_items to orders on order_id
df_join_1 = join_data(order_items, orders, 'left', 'order_id')

# left joins df_join_1 to products on product_id
df_join_2 = join_data(df_join_1, products, 'left', 'product_id')

# left joins df_join_2 to sellers on seller_id
df_join_3 = join_data(df_join_2, sellers, 'left', 'seller_id')

# left joins df_join_3 to prod_name_translator on prod_cat_name
df_join_4 = join_data(df_join_3, prod_name_translator, 'left', 'prod_cat_name')

# left joins df_join_4 to customers on customer_id
df_join_5 = join_data(df_join_4, customers, 'left', 'customer_id')

# left joins df_join_5 to order payment on order_id
olist_data = join_data(df_join_5, order_payment, 'left', 'order_id')

# delete the set of dataframes
del(customers, orders, order_items, order_payment, products, prod_name_translator, sellers)
del(df_join_1, df_join_2, df_join_3, df_join_4, df_join_5)

# check the data types on final dataframe
olist_data.dtypes

# check the column names
olist_data.columns

# --- 4. ETL: data cleaning ---
# check any duplicates
olist_data.duplicated().sum()    

# clean the data -  remove duplicates
olist_data_clean = shuffle(clean_data(olist_data)).reset_index(drop=True)

# subset duplicated rows:
mask_dups = olist_data_clean.duplicated()
df_dups = olist_data_clean[mask_dups]
del(df_dups, olist_data)

# format date into datetime
date_format = '%Y-%m-%d'
convert_dt_as_date(olist_data_clean, 'order_purchase_date', date_format)
convert_dt_as_date(olist_data_clean, 'order_delivered_customer_date', date_format)

# convert data type: object to category
convert_dt_to_cat(olist_data_clean)

################################################
# Part 3 - Exploratory Data Analysis: Insights #
################################################
df_eda = olist_data_clean.copy()
del(olist_data_clean)

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

# define variables: date(s)
vars_dates = ['shipping_limit_date', 'order_purchase_date', 'order_approved_date',
              'order_estimated_delivery_date', 'order_delivered_carrier_date', 
              'order_delivered_customer_date']

# define variables: categorical
vars_cat = df_eda.nunique()[df_eda.nunique() < 28938].keys().tolist()

# remove/add variables to vars_cat
unwanted_items = ['price', 'freight_value', 'order_estimated_delivery_date']
wanted_items = ['order_id', 'customer_id', 'customer_unique_id', 'product_id']

vars_cat = [col for col in vars_cat if col not in unwanted_items]
vars_cat.extend(wanted_items)

# define variables: numerical
vars_num = [col for col in df_eda.columns if col not in vars_dates + vars_cat]

# ---3 perform summary statistics ---
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
df_eda[df_eda.freight_value < 0.915]

# upper bounds (UBs)
df_eda[df_eda.price > 277.4]

# ---5 customer insights ---
# how many orders for this e-commerce dataset
print(df_eda['order_id'].value_counts())

# grouped by 'InvoiceNo' and compute the mean and std.dev
print(df_eda.groupby('order_id').mean())
print(df_eda.groupby('order_id').std())

# grouped by 'order_item_id' to count the transactions
print(df_eda.groupby('order_item_id')['order_id'].value_counts())

# ---5 aggregate dataframe and compute KPIs ---
# Define the dictionary for KPIs:
kpi_dict = {'customer_id':'nunique', 'order_id':'count',
            'price':'mean', 'freight_value':'mean', 
            'payment_value':'mean',}

df_agg_customer = eda_agg_df_var(df_eda, 'customer_unique_id', kpi_dict)

# Print a summary KPI table:
df_agg_customer

# ---6 visualize data: uni-variate ---
# visualize the distribution of 'payment_value', 'freight_value'
sns.distplot(df_eda['payment_value'])
plt.show()

sns.distplot(df_eda['freight_value'])
plt.show()

# plot a histogram:    
plot_hist(df_eda, 'payment_value')

plot_hist(df_eda, 'freight_value')

plot_hist(df_eda, 'price')

# plot a bar chart:
plot_bar_chart(df_eda, 'customer_state')
plot_bar_chart(df_eda, 'prod_cat_name_eng')

# plot a frequency chart:
plot_freq_chart('order_status', 'payment_type', df_eda, None)

# grouped dataframe by a categorical feature:
df_grp_pymt_type = eda_grouped_df_var(df_eda, 'payment_type')
df_grp_order_status = eda_grouped_df_var(df_eda, 'order_status')

# Plot pie chart(s) by categorical features:
plot_pie_chart(df_grp_pymt_type, 'payment_type',
              df_grp_order_status, 'order_status')

# ---7 visualize data: bi-variate ---
# plot a box-whisker:    
plot_box(df_eda, 'payment_value', 'order_status',
        'freight_value', 'order_status',
        'price', 'order_status')

# add 'Country' as a third variable 
sns.boxplot(x='order_status', y='payment_value',
            data=df_eda, sym= '', hue='payment_type')

# create a dataframe:
df_eda_cm = df_eda.copy()

# check data type:
df_eda_cm.info()

# Create a list of variables:
# vars_list = list(df_eda_cm.columns)

# # Delete a list of unwanted variables:
# unwanted_list = ['seller_zip_code_prefix', 'customer_zip_code_prefix', 'order_purchase_date',  
#                   'order_approved_date', 'order_delivered_carrier_date', 'order_delivered_customer_date',
#                   'order_estimated_delivery_date','shipping_limit_date',  'payment_sequential', 'payment_installments',
#                   'prod_cat_name', 'prod_name_length', 'prod_desc_length', 'prod_photo_qty', 'prod_weight_g', 'prod_length_cm',
#                   'prod_height_cm', 'prod_width_cm', 'customer_id', 'customer_unique_id', 'order_id', 'order_item_id',
#                   'product_id', 'seller_id']

# vars_list = [item for item in vars_list if item not in unwanted_list]

# # subset a dataframe
# df_eca_cm = df_eda_cm[vars_list]
    
# # encode categorical variables using Weekly_Sales:
# for col in df_eda_cm.columns:
#     if df_eda_cm[col].dtype.name == 'category':
#         eda_encode_cat_var(df_eda_cm, col, 'payment_value')  

# plot a correlation matrix:
plot_corr_matrix(df_eda_cm, vars_num)

# plot a cross-tabulation:
plot_crosstab(df_eda, 'order_status', 'customer_state')

#---- Plot a scatter plot: w numerical variables ----#
plot_scatter(df_eda, 'price', 'payment_value', 'orange', 100)

plot_scatter(df_eda, 'freight_value', 'payment_value', 'blue', 100)

#---- Plot a linear regression plot: w numerical variables ----#
# Compute Pearson r for combination of X and Y:
r_freight_payment = compute_pearson_r(df_eda, 'freight_value', 'payment_value')
r_price_payment = compute_pearson_r(df_eda, 'price', 'payment_value')

# Plot a linear regression analysis:
plot_linear_reg(df_eda, 'freight_value', 'payment_value',
                r_freight_payment, 'green', 'Transactions')
plt.show()

# Plot a linear regression analysis:
plot_linear_reg(df_eda, 'price', 'payment_value',
                r_price_payment, 'blue', 'Transactions')
plt.show()

# specify the output_path
output_path = r'D:/TY Workspace/3. Data Science Portfolio/1. Completed Portfolios/P6_Customer_Segments_Oilst/2_Development/Output'

# save model results as .csv file:
# write_as_csv(results, 'pred_results.csv', output_path)