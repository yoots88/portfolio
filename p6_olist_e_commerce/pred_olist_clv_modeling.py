# Import and set directory:
import os
os.getcwd()

# Set path:
input_path = 'D:/TY Workspace/3. Data Science Portfolio/1. Completed Portfolios/P6_Customer_Segments_Oilst/2_Development/Input'
os.chdir(input_path)

# Import all libraries #
import sys
import re
import numpy as np
import pandas as pd
import csv
from datetime import datetime, date, timedelta 
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
import statsmodels.api as sm
import scipy.stats as sp
import sklearn.metrics as skm
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler, PolynomialFeatures 
from sklearn.model_selection import KFold, train_test_split
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Authorship:
__author__ = 'Taesun Yoo'
__email__ = 'yoots1988@gmail.com'

# Check out Python version:
print ("Python version: {}".format(sys.version))

#############################
# Part 2 - DISCOVER PHASE ###
#############################
# --- 1. Write Out List of Functions --- #
def load_file(file):
    '''load input CSVs as a dataframe'''
    return pd.read_csv(file, encoding='latin1')


def convert_dt_as_custom(df, var_name, dt_type):
    '''convert datatype on selected variables'''
    df[var_name] = df[var_name].astype(dt_type)
    return df[var_name]


def convert_dt_as_category(df):
    '''convert datatype from object to category'''
    for col in df.columns:
        if df[col].dtype.name == 'object':
            df[col] = df[col].astype('category')


def convert_dt_as_date(df, var_name, date_format):
    '''convert the column as date format'''
    df[var_name] = pd.to_datetime(df[var_name], format=date_format)


def convert_dt_as_date_object(df, col):
    '''convert datetime to object with parse date only'''
    df[col] = df[col].dt.date
    return df[col]


def join_data(df1, df2, join_type, key=None,
              left_index=None, right_index=None):
    '''merge the dataframes by a key'''
    df_join = pd.merge(df1, df2, how=join_type, on=key,
                       left_index=False, right_index=False)
    return df_join


def clean_data(raw_df):
    '''remove rows that contain duplicate columns'''
    clean_df = raw_df.drop_duplicates()
    # clean_df = raw_df.drop_duplicates(subset='customer_id')
    # clean_df = clean_df[clean_df.var_X > 0]
    # clean_df = clean_df[clean_df.var_Z <= 11000]
    return clean_df


def drop_data(df, var_list):
    '''drop variables from a dataframe'''
    df = df.drop(var_list, axis=1)
    return df


def EDA_missing_data(df):
    ''' compute missing value % on a df'''
    df_missing = pd.DataFrame(df.isnull().sum())
    df_missing.columns = ['count']
    df_missing = df_missing.sort_values(by='count', ascending=False)
    df_missing['pct'] = (df_missing['count']/len(df)) * 100
    return df_missing


def EDA_summary_stat_num(df):
    ''' compute numerical summary statistics '''
    df_stat_num = df.describe().T
    df_stat_num = df_stat_num[['count', 'min', 'mean', 'max', '25%', '50%', '75%', 'std']]
    df_stat_num = df_stat_num.sort_values(by='count', ascending=True)
    df_stat_num = pd.DataFrame(df_stat_num)
    return df_stat_num


def EDA_summary_stat_cat(df):
    ''' compute numerical summary statistics '''
    df_stat_cat = pd.DataFrame(df.describe(include='category').T)
    # df_stat_cat = pd.DataFrame(df.describe(include='O').T)
    return df_stat_cat


def feature_replacement(df):
    ''' replace missing values based on specific data type of a column '''
    for col in df.columns:
        if df[col].dtype.name == 'object':
            mode = df[col].mode().iloc[0]
            df[col] = df[col].fillna(mode)
        elif df[col].dtype.name == 'float64':
            mean = df[col].mean()
            df[col] = df[col].fillna(mean)
        elif df[col].dtype.name == 'datetime64[ns]':
            # pseudo_date = max(df[col])
            pseudo_date = pd.Timestamp.max
            df[col] = df[col].fillna(pseudo_date)
        else:
            df[col].dtype.name == 'int64'
            median = df[col].median()
            df[col] = df[col].fillna(median)


def get_dummies(df, var_name):
    '''Create a list of dummy vars'''
    dummies = pd.get_dummies(df[var_name], drop_first=True)
    return dummies    


def model_tuning_param(model, feature_df, label_df, param_dist, n_iter):
    '''Performs RandomizedSearchCV to tune model parameters'''
    random_search = RandomizedSearchCV(model, param_dist, n_iter, cv=5)
    random_search.fit(feature_df, label_df)
    return random_search


def print_best_params(random_search, param_1=None, param_2=None, param_3=None):
    '''Print the best model parameter'''
    print("Best " + param_1 + ":", random_search.best_estimator_.get_params()[param_1])
    print("Best " + param_2 + ":", random_search.best_estimator_.get_params()[param_2])
    print("Best " + param_3 + ":", random_search.best_estimator_.get_params()[param_3])


def model_train(model, X_train, y_train, n_proc, mean_mae, cv_std):
    '''Train a model and output mean MAE and CV Std. MAE'''
    #scoring = 'neg_mean_absolute_error', 'neg_mean_squared_error'
    # dont forget np.sqrt() for RMSE metric!
    neg_mae = cross_val_score(model, X_train, y_train, n_jobs=n_proc, cv=5,
                              scoring='neg_mean_absolute_error')
    mae = -1*neg_mae
    mean_mae[model] = np.mean(mae)
    cv_std[model] = np.std(mae)

def model_summary(model, mean_mae, cv_std):
    '''Print out the model performance summary'''
    print('\nModel:\n', model)
    print('Average MAE:\n', mean_mae[model])
    print('Std. Dev during CV:\n', cv_std[model])

    
def model_results(model, mean_mae, predictions, feature_importances):
    '''Saves the model name, mean_mae, predicted sales, and feature importances'''
    with open('model.txt', 'w') as file:
        file.write(str(model))
        feature_importances.to_csv('feat_importances.csv',index=False)
        predictions.to_csv('pred_results_best.csv',index=False,index_label=None)


def plot_reg_compare(y_train_pred, y_train_act):
    '''Plot a scatter plot to compare predicted vs actual label'''
    plt.scatter(y_train_pred, y_train_act, c='blue', 
                edgecolor='w', marker='o', label='train set')
    plt.xlabel('Predicted purchased items')
    plt.ylabel('Actual purchased items')
    plt.legend(loc='upper left')


def plot_reg_residual(y_train_pred, y_train_act):
    '''Plot a scatter plot to visualize residual from predicted vs. actual label'''
    plt.scatter(y_train_pred, (y_train_pred - y_train_act), c='blue',
                edgecolor='w', marker='o', label='train set')
    plt.hlines(y=0, xmin=0, xmax=max(y_train_act), colors='k', lw=3)
    plt.xlim([0, max(y_train_act)])
    plt.xlabel('Predicted purchased items')
    plt.ylabel('Residual purchased items')
    plt.legend(loc='upper left')    


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

# --- 4. ETL data cleaning ---
# define unwanted variables
vars_unwanted = ['seller_zip_code_prefix', 'customer_zip_code_prefix', 'order_approved_date', 'order_delivered_carrier_date',
                 'order_estimated_delivery_date','shipping_limit_date',  'payment_sequential', 'payment_installments',
                 'prod_name_length', 'prod_desc_length', 'prod_photo_qty', 'prod_weight_g', 'prod_length_cm',
                 'prod_height_cm', 'prod_width_cm']

vars_dates = ['order_purchase_date', 'order_delivered_customer_date']

# drop the data - unwanted variables
olist_data = drop_data(olist_data, vars_unwanted)
del(vars_unwanted)

# clean the data -  remove duplicates
olist_data_clean = shuffle(clean_data(olist_data)).reset_index(drop=True)

del(olist_data)

# define categorical
vars_cat = olist_data_clean.nunique()[olist_data_clean.nunique() < 28938].keys().tolist()

# from vars_cat: remove or add variables 
unwanted_items = {'price', 'freight_value'}
wanted_items = {'order_id', 'customer_id', 'customer_unique_id', 'product_id'}

# remove unwated items
vars_cat = [col for col in vars_cat if col not in unwanted_items]

# append the wanted items
vars_cat.extend(wanted_items)

# define numerical
vars_num = [col for col in olist_data_clean.columns if col not in vars_dates + vars_cat]
del(unwanted_items, wanted_items)

# format date into datetime
date_format = '%Y-%m-%d'
convert_dt_as_date(olist_data_clean, 'order_purchase_date', date_format)
convert_dt_as_date(olist_data_clean, 'order_delivered_customer_date', date_format)

# parse only dates
olist_data_clean['order_purchase_date'] = olist_data_clean['order_purchase_date'].dt.floor('d')
olist_data_clean['order_delivered_customer_date'] = olist_data_clean['order_delivered_customer_date'].dt.floor('d')

# subset dataframe where olist_data_clean for "prod_cat_name_eng" is not na
olist_data_clean = olist_data_clean[olist_data_clean.prod_cat_name_eng.notna()]

# --- 5. Feature Imputation ---
# before imputation
df_missing_pre = EDA_missing_data(olist_data_clean)
df_missing_pre

# feature imputation
feature_replacement(olist_data_clean)

# after imputation
df_missing_post = EDA_missing_data(olist_data_clean)
df_missing_post

del(df_missing_pre, df_missing_post)

# --- 6. EDA: numerical and categorical ---
olist_data_clean.info()

# convert the order_item_id as object
convert_dt_as_custom(olist_data_clean, 'order_item_id', 'object')
 
# convert data type: object to category
convert_dt_as_category(olist_data_clean)

# compute summary stat. by numerical
df_stat_num = EDA_summary_stat_num(olist_data_clean[vars_num])
df_stat_num    

# compute summary stat. by categorical
df_stat_cat = EDA_summary_stat_cat(olist_data_clean[vars_cat])
df_stat_cat 

# --- 6. Cohort Analysis --- #
# Monthly Grouping #
def get_month(x): 
    '''return year-month value from order_purchase_date'''
    return date(x.year, x.month, 1) 

# Create PurchaseMonth column
olist_data_clean['PurchaseMonth'] = olist_data_clean['order_purchase_date'].apply(get_month) 

# Group by CustomerID and select the PurchaseMonth value
grouping_month = olist_data_clean.groupby('customer_unique_id')['PurchaseMonth'] 

# Assign a minimum PurchaseMonth value as the CohortMonth
olist_data_clean['CohortMonth'] = grouping_month.transform('min')

# View the top 5 rows
print(olist_data_clean.head())

# fix the data types on: PurchaseMonth, CohortMonth
olist_data_clean['PurchaseMonth'] = pd.to_datetime(olist_data_clean['PurchaseMonth'], errors='coerce', format=date_format)
olist_data_clean['CohortMonth'] = pd.to_datetime(olist_data_clean['CohortMonth'], errors='coerce', format=date_format)

del(grouping_month)

# check data types:
olist_data_clean.dtypes

# --- calculate time offset in days - part 1 ---
def get_date_int(df, column):
    year = df[column].dt.year
    month = df[column].dt.month
    return year, month

# Get the integers for date parts from the `PurchaseMonth` column
purchase_year, purchase_month = get_date_int(olist_data_clean, 'PurchaseMonth')

# Get the integers for date parts from the `CohortMonth` column
cohort_year, cohort_month = get_date_int(olist_data_clean, 'CohortMonth')

# --- calculate time offset in days - part 2 ---
# Calculate difference in years
diff_years = purchase_year - cohort_year

# Calculate difference in months
diff_months = purchase_month - cohort_month

# Extract the difference in days from all previous values
olist_data_clean['CohortIndex_Month'] = diff_years * 12 + diff_months + 1

print(olist_data_clean.head())

del(cohort_year, purchase_year, diff_years)
del(cohort_month, purchase_month, diff_months)

# format PurchaseDay, CohortDay, PurchaseMonth, CohortMonth back to to object:
olist_data_clean['PurchaseMonth'] = convert_dt_as_date_object(olist_data_clean, 'PurchaseMonth')
olist_data_clean['CohortMonth'] = convert_dt_as_date_object(olist_data_clean, 'CohortMonth')
    
# --- 7. CLV Model: Aggregate by Population ---
# group customers by customer's unique id
group_by_customer = olist_data_clean.groupby(
    by = olist_data_clean['customer_unique_id'],
    as_index = False)

# calculate duration bewteen first to last order purchase by customers
customers = group_by_customer['order_purchase_date'] \
    .agg({'order_purchase_date': lambda x: (x.max()-x.min())}).rename(columns={'order_purchase_date':'num_days'})

# calculate customer's tenure period in months:
customers['tenure_months'] = np.maximum(customers["num_days"] \
    .apply(lambda x: np.ceil(x.days / 30)), 1.0)

# check top 5 rows
customers.head()

# define the churning
twelve_weeks = timedelta(weeks=12)
cutoff_date = olist_data_clean['order_purchase_date'].max()

# determine whether a customer is inactive after 12 weeks
dead = group_by_customer['order_purchase_date'].max()['order_purchase_date']\
    .apply(lambda x: (cutoff_date - x) > twelve_weeks)

# calculate the customer churn ratio
churn = dead.sum() / customers['tenure_months'].sum()

# calculate net spend over customer's tenure months
spend = olist_data_clean['payment_value'].sum() / customers['tenure_months'].sum()

# calculate the overall CLV by aggregate model
clv_model_aa = spend/churn

# print the value of Aggregate CLV model
print("CLV in USD:{0:4f}".format(clv_model_aa * 0.19))

# --- 8. CLV Model: Aggregate by Customer ---
# append the total spend by customer id
customers_ac = customers.merge(
    group_by_customer['payment_value'].sum(),
    on = 'customer_unique_id')

# calculate the CLV per customer
customers_ac['CLV'] = (customers_ac['payment_value']/customers_ac['tenure_months'])/churn

# check the top 5 rows
customers_ac.head()

# plot the histogram: Analytic CLV by Cohort
plt.hist(customers_ac['CLV']*0.19, range = (0, 600))
plt.title('Analytical Cohort CLV')
plt.xlabel('CLV ($)')
plt.ylabel('# of Customers')
plt.savefig('hist_olist_cohort_CLV_AC.png', dpi=300, bbox_inches='tight')
plt.show()

# specify the output_path
output_path = r'D:/TY Workspace/3. Data Science Portfolio/1. Completed Portfolios/P6_Customer_Segments_Oilst/2_Development/Output'

# save model results as .csv file:
# write_as_csv(results, 'pred_unique_items.csv', output_path)