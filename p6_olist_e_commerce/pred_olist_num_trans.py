# Import and set directory:
import os
os.getcwd()

# Set path:
input_path = 'D:/TY Workspace/3. Data Science Portfolio/1. Completed Portfolios/P6_Customer_Segments_Oilst/2_Development/Input'
os.chdir(input_path)
    
# Import all libraries #
from datetime import datetime, date, timedelta
import sys
import numpy as np
import pandas as pd
import csv
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


def feature_replacement(X):
    ''' replace missing values based on specific data type of a column '''
    for col in X.columns:
        if X[col].dtype.name == 'object':
            mode = X[col].mode().iloc[0]
            X[col] = X[col].fillna(mode)
        elif X[col].dtype.name == 'float64':
            mean = X[col].mean()
            X[col] = X[col].fillna(mean)
        elif X[col].dtype.name == 'datetime64[ns]':
            pseudo_date = pd.Timestamp.max
            X[col] = X[col].fillna(pseudo_date)
        else:
            X[col].dtype.name == 'int64'
            median = X[col].median()
            X[col] = X[col].fillna(median)

def format_dummy_vars(df, var_name):
    '''format dummy variables in recognized names'''
    df[var_name] = var_name + '_' + df[var_name].map(str)


def get_dummies(df, var_name):
    '''Create a list of dummy vars'''
    dummies = pd.get_dummies(df[var_name], drop_first=True)
    return dummies    


def split_dataframe(df):
    '''Split data into train and test sets'''
    df_final_train, df_final_test = df.loc[df['File_Type']=='Train'], df.loc[df['File_Type']=='Test']
    return df_final_train, df_final_test


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


def gen_predictions_file(df, var_1, var_2, var_3):
    '''Generate test predictions file: concatenate with 3 keys to make the id column'''
    df['id'] = df[var_1].map(str)+'_'+df[var_2].map(str)+'_'+df[var_3].map(str)
    return df[['id', 'Weekly_Sales_Pred']]


def plot_reg_compare(y_train_pred, y_train_act):
    '''Plot a scatter plot to compare predicted vs actual label'''
    plt.scatter(y_train_pred, y_train_act, c='blue', 
                edgecolor='w', marker='o', label='train set')
    plt.xlabel('Predicted transactions')
    plt.ylabel('Actual transactions')
    plt.legend(loc='upper left')


def plot_reg_residual(y_train_pred, y_train_act):
    '''Plot a scatter plot to visualize residual from predicted vs. actual label'''
    plt.scatter(y_train_pred, (y_train_pred - y_train_act), c='blue',
                edgecolor='w', marker='o', label='train set')
    plt.hlines(y=0, xmin=0, xmax=max(y_train_act), colors='k', lw=3)
    plt.xlim([0, max(y_train_act)])
    plt.xlabel('Predicted transactions')
    plt.ylabel('Residual transactions')
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


##############################
# --- 6. Cohort Analysis --- #
##############################
# Daily Grouping #
def get_day(x): 
    '''return year-month-day value from order_purchase_date'''
    return date(x.year, x.month, x.day) 

# Create PurchaseDay column
olist_data_clean['PurchaseDay'] = olist_data_clean['order_purchase_date'].apply(get_day) 

# Group by CustomerID and select the InvoiceDay value
grouping_day = olist_data_clean.groupby('customer_unique_id')['PurchaseDay'] 

# Assign a minimum InvoiceDay value to the dataset
olist_data_clean['CohortDay'] = grouping_day.transform('min')

# fix the data types on: InvoiceDay, CohortDay
olist_data_clean['PurchaseDay'] = pd.to_datetime(olist_data_clean['PurchaseDay'], errors='coerce', format=date_format)
olist_data_clean['CohortDay'] = pd.to_datetime(olist_data_clean['CohortDay'], errors='coerce', format=date_format)

del(grouping_day)

# Monthly Grouping #
def get_month(x): 
    '''return year-month value from order_purchase_date'''
    return date(x.year, x.month, 1) 

# Create PurchaseMonth column
olist_data_clean['PurchaseMonth'] = olist_data_clean['order_purchase_date'].apply(get_month) 

# Group by CustomerID and select the PurchaseMonth value
grouping_month = olist_data_clean.groupby('customer_unique_id')['PurchaseMonth'] 

# Assign a minimum PurchaseMonth value to the dataset
olist_data_clean['CohortMonth'] = grouping_month.transform('min')

# View the top 5 rows
print(olist_data_clean.head())

# fix the data types on: PurchaseMonth, CohortMonth
olist_data_clean['PurchaseMonth'] = pd.to_datetime(olist_data_clean['PurchaseMonth'], errors='coerce', format=date_format)
olist_data_clean['CohortMonth'] = pd.to_datetime(olist_data_clean['CohortMonth'], errors='coerce', format=date_format)

del(grouping_month)

# Yearly Grouping #
def get_year(x): 
    '''return year value from order_purchase_date'''
    return datetime(x.year, 1, 1)

# Create PurchaseMonth column
olist_data_clean['PurchaseYear'] = olist_data_clean['order_purchase_date'].apply(get_year) 

# Group by CustomerID and select the PurchaseYear value
grouping_year = olist_data_clean.groupby('customer_unique_id')['PurchaseYear'] 

# Assign a minimum PurchaseYear value to the dataset
olist_data_clean['CohortYear'] = grouping_year.transform('min')

# View the top 5 rows
print(olist_data_clean.head())

# fix the data types on: PurchaseYear, CohortYear
olist_data_clean['PurchaseYear'] = pd.to_datetime(olist_data_clean['PurchaseYear'], errors='coerce', format=date_format)
olist_data_clean['CohortYear'] = pd.to_datetime(olist_data_clean['CohortYear'], errors='coerce', format=date_format)

del(grouping_year)

# check data types:
olist_data_clean.dtypes

# --- calculate time offset in days - part 1 ---
def get_date_int(df, column):
    year = df[column].dt.year
    month = df[column].dt.month
    day = df[column].dt.day
    return year, month, day

# Get the integers for date parts from the `PurchaseDay`, `PurchaseMonth` column
purchase_year, purchase_month, purchase_day = get_date_int(olist_data_clean, 'PurchaseDay')

# Get the integers for date parts from the `CohortDay`, `CohortMonth` column
cohort_year, cohort_month, cohort_day = get_date_int(olist_data_clean, 'CohortDay')

# --- calculate time offset in days - part 2 ---
# Calculate difference in years
diff_years = purchase_year - cohort_year

# Calculate difference in months
diff_months = purchase_month - cohort_month

# Calculate difference in days
diff_days = purchase_day - cohort_day

# Extract the difference in days from all previous values
olist_data_clean['CohortIndex_Year'] = diff_years + 1
olist_data_clean['CohortIndex_Month'] = diff_years * 12 + diff_months + 1
olist_data_clean['CohortIndex_Day'] = diff_years * 365 + diff_months * 30 + diff_days + 1

print(olist_data_clean.head())

del(cohort_year, cohort_month, cohort_day)
del(purchase_year, purchase_month, purchase_day)
del(diff_years, diff_months, diff_days)

# format PurchaseDay, CohortDay, PurchaseMonth, CohortMonth back to to object:
olist_data_clean['PurchaseDay'] = convert_dt_as_date_object(olist_data_clean, 'PurchaseDay')
olist_data_clean['CohortDay'] = convert_dt_as_date_object(olist_data_clean, 'CohortDay')
olist_data_clean['PurchaseMonth'] = convert_dt_as_date_object(olist_data_clean, 'PurchaseMonth')
olist_data_clean['CohortMonth'] = convert_dt_as_date_object(olist_data_clean, 'CohortMonth')
olist_data_clean['PurchaseYear'] = convert_dt_as_date_object(olist_data_clean, 'PurchaseYear')
olist_data_clean['CohortYear'] = convert_dt_as_date_object(olist_data_clean, 'CohortYear')

# --- calculate average payment ---
# Group customers based on the cohort month and index month
monthly_grouping = olist_data_clean.groupby(['CohortMonth','CohortIndex_Month'])

# Calculate the average of the unit price column
monthly_cohort_payment = monthly_grouping['payment_value'].mean()

# Reset the index of cohort_data
monthly_cohort_payment = monthly_cohort_payment.reset_index()

# Create a pivot 
monthly_average_payment = monthly_cohort_payment.pivot(index='CohortMonth', 
                                                       columns='CohortIndex_Month',
                                                       values='payment_value')

del(monthly_cohort_payment)

# --- visualize average payment by monthly cohorts ---
# Initialize an 8 by 6 inches plot figure
plt.figure(figsize=(15, 10))

# Add a title
plt.title('Average Payment by Monthly Cohorts')

# Create the heatmap
sns.heatmap(monthly_average_payment.round(1), annot=True, cmap='Blues', fmt='g')
plt.show()

# --- calculate retention rate from scratch ---
# Count the number of unique values per customer_unique_id
monthly_cohort_data = monthly_grouping['customer_unique_id'].apply(pd.Series.nunique).reset_index()

# Convert the CohortMonth as object:
convert_dt_as_date_object(monthly_cohort_data, 'CohortMonth')

# Create a pivot 
monthly_cohort_counts = monthly_cohort_data.pivot(index='CohortMonth', columns='CohortIndex_Month', values='customer_unique_id')

# Extract cohort sizes from the first column of cohort_counts
monthly_cohort_sizes = monthly_cohort_counts.iloc[:,0]

# Calculate retention by dividing the counts with the cohort sizes
monthly_retention = monthly_cohort_counts.divide(monthly_cohort_sizes, axis=0)

del(monthly_cohort_data, monthly_cohort_counts, monthly_cohort_sizes)

# Calculate churn
monthly_churn = 1 - monthly_retention

# --- visualize retention rate by monthly cohorts  ---
# Initialize an 8 by 6 inches plot figure
plt.figure(figsize=(15, 10))

# Add a title
plt.title('Retention Rate by Monthly Cohorts')

# Create the heatmap
sns.heatmap(monthly_retention.round(3), annot=True, cmap='Blues', fmt='g')
plt.show()

#############################################
# --- 7. Customer Lifetime Value (CLV) --- #
############################################
# --- explore retention and churn ---
# exclude the first month values as they are constant
# given this is the first month the customers have been active
# chaining: monthly mean --> overall mean rates

# Calculate the mean retention rate
retention_rate = monthly_retention.iloc[:,1:].mean().mean()

# Calculate the mean churn rate
churn_rate = monthly_churn.iloc[:,1:].mean().mean()

# Print rounded retention and churn rates
print('Retention rate: {:.2f}; Churn rate: {:.2f}'.format(retention_rate, churn_rate))

# --- calculate basic CLV ---
# Calculate monthly spend per customer
monthly_revenue = olist_data_clean.groupby(['customer_unique_id','PurchaseMonth'])['payment_value'].sum()

# Calculate average monthly spend
monthly_revenue = np.mean(monthly_revenue)

# Define lifespan to 21 months based on the order purchase period
lifespan_months = 21

# Calculate basic CLV
clv_basic = monthly_revenue * lifespan_months

# Print the basic CLV value: currency conversion 1 Real = 0.26 CAD
print('Average basic CLV is {:.1f} CAD'.format(clv_basic * 0.26))

# --- calculate granular CLV ---
# Calculate average revenue per invoice
revenue_per_purchase = olist_data_clean.groupby(['order_id'])['payment_value'].mean().mean()

# Calculate average number of unique orders per customer per month
frequency_per_month = olist_data_clean.groupby(['customer_unique_id','PurchaseMonth'])['order_id'].nunique().mean()

# Calculate granular CLV
clv_granular = revenue_per_purchase * frequency_per_month * lifespan_months

# Print the granular CLV value: currency conversion 1 Real = 0.26 CAD
print('Average granular CLV is {:.1f} CAD'.format(clv_granular * 0.26))

# --- calculate traditional CLV ---
# Calculate monthly spend per customer
monthly_revenue = olist_data_clean.groupby(['customer_unique_id','PurchaseMonth'])['payment_value'].sum().mean()

# Calculate average monthly retention rate
retention_rate = monthly_retention.iloc[:,1:].mean().mean()

# Calculate average monthly churn rate
churn_rate = 1 - retention_rate

# Calculate traditional CLV 
clv_traditional = monthly_revenue * (retention_rate / churn_rate)

# Print traditional CLV and the retention rate values: currency conversion 1 Real = 0.26 CAD
print('Average traditional CLV is {:.1f} CAD at {:.1f} % retention_rate'.format(clv_traditional * 0.26, retention_rate*100))

# --- define a set of features ---
# Define the snapshot date as the latest PurchaseMonth
snapshot_dt = datetime(2018,8,1)

# Explore the monthly distribution of transactions
olist_data_clean.groupby(['PurchaseMonth']).size()

# Exclude target variable
olist_data_clean_X = olist_data_clean[olist_data_clean['order_purchase_date'].dt.strftime('%Y-%m') != '2018-08']

# Prepare the features dataframe based on unique CustomerID
features = olist_data_clean_X.groupby('customer_unique_id').agg({
  'order_purchase_date': lambda x: (snapshot_dt - x.max()).days,
  # Calculate frequency by counting unique number of orders
  'order_id': pd.Series.nunique,
  # Calculate monetary value by summing all payment values
  'payment_value': np.sum,
  # Calculate number of order items
  'order_item_id': 'count',
  # Calculate number of products
  'product_id': 'count'}).reset_index()

# Rename the columns
features.columns = ['CustomerID', 'recency', 'frequency', 'monetary', 'items_count', 'products_count']

# Replace NaN with 0 in this context
features.recency = np.where(features.recency.isnull(), 0, features.recency)

# --- define a target variable ---
# Build a pivot table with monthly orders per customer
cust_month_orders = pd.pivot_table(data=olist_data_clean, values='order_id',
                               index=['customer_unique_id'], columns=['PurchaseMonth'],
                               aggfunc=pd.Series.nunique, fill_value=0)

# Delete '2019-09-01' data from cust_month_orders
cust_month_orders = cust_month_orders.iloc[:, :-1]

# Store August 2018 data column name as a list
target = [pd.to_datetime('2018-08-01').date()]

# Store target value as `Y`
Y = cust_month_orders[target]

# --- split data into training and test sets ---
# Store customer identifier column name as a list
custid = 'CustomerID'

# Select feature column names excluding customer identifier
cols = [col for col in features.columns if col not in custid]

# Split data to training and testing
X_train, X_val, y_train, y_val = train_test_split(features, Y, test_size=0.25, random_state=99)
    
# Subset customer_id for X_val: 
X_cust_id_val = X_val[custid]

# Extract the features on X_train and X_val:
X_train, X_val  = X_train[cols], X_val[cols]

# --- predict next month orders ---
# Create a model instance
baseline = LinearRegression()

# Compute mean absolute error
baseline_cv_mae = cross_val_score(baseline, X_train, y_train,
                              cv=5, scoring='neg_mean_absolute_error')
baseline_mae = -1 * baseline_cv_mae
baseline_mean_mae = np.mean(baseline_mae)

# Compute mean root mean squared error
baseline_cv_mse = cross_val_score(baseline, X_train, y_train,
                              cv=5, scoring='neg_mean_squared_error')
baseline_rmse = np.sqrt(-1 * baseline_cv_mse)
baseline_mean_rmse  = np.mean(baseline_rmse)

print('Baseline MAE: {}; Baseline RMSE: {}'.format(str(baseline_mean_mae), str(baseline_mean_rmse)))

# --- 15. Create models --- # 
# initialize model list and dicts
models = []
mean_mae = {}
cv_std = {}
res = {}

# define common model parameters: num processors and shared model parameters
n_proc = 1
verbose_lv = 5

# # create and tune the models that you brainstormed during part 2
# ###############################################################################        
# # Hyper-parameters tuning: RandomForest
# rf = RandomForestRegressor()   
# n_iter = 10
# param_dist_rf = {'n_estimators':sp.randint(10,50), 
#                   'max_depth':sp.randint(1,10),
#                   'min_samples_split':sp.randint(10,60)}

# random_search_rf = model_tuning_param(rf, X_train, y_train, param_dist_rf, n_iter)    
    
# # print the best model parameters: RandomForest    
# param_1 = 'n_estimators' 
# param_2 = 'max_depth'
# param_3 = 'min_samples_split'
    
# print_best_params(random_search_rf, param_1, param_2, param_3)
# ###############################################################################    

# # Hyper-parameters tuning: GradientBoosting
# gbr = GradientBoostingRegressor()
# n_iter = 10
# param_dist_gbr = {'n_estimators':sp.randint(10,40), 
#                   'max_depth':sp.randint(1,20),
#                   'loss':['ls']}

# random_search_gbr = model_tuning_param(gbr, X_train, y_train, param_dist_gbr, n_iter)    
    
# # print the best model parameters: GradientBoosting    
# param_1 = 'n_estimators' 
# param_2 = 'max_depth'
# param_3 = 'loss'
    
# print_best_params(random_search_gbr, param_1, param_2, param_3)        
# ###############################################################################    

# # Hyper-parameters tuning: XGBoost
# xgb = XGBRegressor()   
# n_iter = 10
# param_dist_xgb = {'n_estimators':sp.randint(10,40), 
#                   'max_depth':sp.randint(1,20),
#                   'learning_rate':np.random.uniform(0,1,10)}

# random_search_xgb = model_tuning_param(xgb, X_train, y_train, param_dist_xgb, n_iter)    
    
# # print the best model parameters: XGBRegressor    
# param_1 = 'n_estimators' 
# param_2 = 'max_depth'
# param_3 = 'learning_rate'
    
# print_best_params(random_search_xgb, param_1, param_2, param_3)

# --- 16. Cross-validate models --- # 
# do 2-fold cross validation on models and measure MAE
# Model List to train: Order of Model Complexity
lr_L1 = Lasso(alpha=0.01)
lr_L2 = Ridge(alpha=0.01)
rf = RandomForestRegressor(n_estimators=32, n_jobs=n_proc, max_depth=5,
                               min_samples_split=26, verbose=verbose_lv)   
gbr = GradientBoostingRegressor(n_estimators=29, max_depth=2, loss='ls', verbose=verbose_lv)
xgb = XGBRegressor(n_estimators=23, max_depth=2, learning_rate=0.5436747777101026) 

# a list of regressors:
models.extend([lr_L1, lr_L2, rf, gbr, xgb])    

# cross-validate models, using MAE to evaluate and print the summaries
print("begin cross-validation")
for model in models:
    model_train(model, X_train, y_train, n_proc, mean_mae, cv_std)
    model_summary(model, mean_mae, cv_std)

# --- 17. Select the best model with lowest MAE for your prediction model --- #
# Best Model XGB: MAE = , CV_STD_MAE =     
model = min(mean_mae, key=mean_mae.get)
print('\nBest model with the lowest MAE:')
print(model)

# --- 18. Model Evaluation: Scatter Plot --- #
# re-train a model with best model:
model.fit(X_train, y_train)
y_train_pred = model.predict(X_train).reshape(len(y_train),1)
y_train_pred = pd.DataFrame(y_train_pred, columns=['2018-08-01'])

# Plot a comparison scatter plot: predicted vs. actual
plt.figure(figsize=(14,7))
plt.subplots_adjust(wspace=0.2)
plt.subplot(1,2,1)
plot_reg_compare(y_train_pred.values, y_train.values)
plt.title('Transactions count predicted vs. actual')

# Plot a residual scatter plot: predicted vs. actual
plt.subplot(1,2,2)
plot_reg_residual(y_train_pred.values, y_train.values)
plt.title('Transactions count predicted vs. actual')
plt.show()

# --- 19. Save Prediction Results ---
# make predictions on a validation set
df_pred = model.predict(X_val)

# save it as a dataframe:
results = pd.DataFrame({'customer_unique_id': X_cust_id_val,
                      'unique_orders':df_pred})

# specify the output_path
output_path = r'D:/TY Workspace/3. Data Science Portfolio/1. Completed Portfolios/P6_Customer_Segments_Oilst/2_Development/Output'

# save model results as .csv file:
write_as_csv(results, 'pred_unique_orders_rf.csv', output_path)