# Import and set directory:
import os
os.getcwd()

# Set input path
input_path = 'D:/Virtual_Internships/1_KPMG_Data Analytics/Input'
os.chdir(input_path)

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
def load_file_excel(file, sheet_name):
    '''load input XLSs as a dataframe'''
    return pd.read_excel(file, sheet_name=sheet_name)


def drop_data(df, var_list):
    ''' drop variables based on the specified list '''
    df = df.drop(var_list, axis=1)
    return df


def drop_duplicates_data(data, col_id=None):
    '''subset data with non-duplicates'''
    # sort by a specified id
    data.sort_values(col_id, inplace=True)
    # make a bool series for duplicates
    bool_series_dups = data.duplicated(keep=False)
    # subset a dataframe with non-duplicate
    data_clean = data[~bool_series_dups]
    # return a subset dataframe
    return data_clean

         
def convert_dt_as_cat(df):
    ''' convert data type object to category '''
    for col in df.columns:
        if df[col].dtype.name == 'object':
            df[col] = df[col].astype('category')
            

def convert_int_to_datetime(df, col):
    '''convert integer to datetime'''
    df[col] = pd.to_datetime(df[col], unit='d', origin='1900-01-01', errors='coerce')
    return df[col]


def join_data(df1, df2, join_type, key=None,
              left_index=None, right_index=None):
    '''merge the dataframes by a key'''
    df_join = pd.merge(df1, df2, how=join_type, on=key,
                       left_index=False, right_index=False)
    return df_join


def eda_missing(df):
    '''compute missing % on each var'''
    df_missing = pd.DataFrame(df.isnull().sum(), columns=['count'])
    df_missing['pct'] = (df_missing['count']/len(df)) * 100
    return df_missing


def eda_stat_cat(df):
    ''' perform eda for categorical features '''
    df_stat_cat = df.describe(include='category').T
    df_stat_cat = pd.DataFrame(df_stat_cat)
    return df_stat_cat


def eda_stat_num(df):
    ''' perform eda for numerical features '''
    df_stat_num = df.describe().T
    df_stat_num = df_stat_num[['count', 'min', 'mean', 'max', 'std', '25%', '50%', '75%']]
    df_stat_num = pd.DataFrame(df_stat_num)
    return df_stat_num


def eda_outliers(df):
    '''check outliers using the IQR method'''
    df['IQR'] = df['75%'] - df['25%']
    df['LB']  = df['25%'] - 1.5*df['IQR']
    df['UB']  = df['75%'] + 1.5*df['IQR']
    df = df.drop(['count','std','mean','25%','50%','75%','IQR'], axis=1)
    return df


def list_comparator(l2, l1):
    ''' compare the two lists and return the difference '''
    diff_list = list(set(l2) - set(l1))
    return diff_list


def gender_conditions(df):
    '''return a list with series'''
    conditions = [df['gender'] == 'F', 
              df['gender'] == 'Female',
              df['gender'] == 'Femal',
              df['gender'] == 'U',
              df['gender'] == 'M',
              df['gender'] == 'Male']
    return conditions


def deceased_conditions(df):
    '''return a list with series'''
    conditions = [df['deceased_indicator'] == 'Y', 
              df['deceased_indicator'] == 'N']
    return conditions


def order_conditions(df):
    '''return a list with series'''
    conditions = [df['online_order'] == 1, 
              df['online_order'] == 0]
    return conditions


def state_conditions(df):
    '''return a list with series'''
    conditions = [df['state'] == 'New South Wales', 
              df['state'] == 'NSW',
              df['state'] == 'Victoria', 
              df['state'] == 'QLD',
              df['state'] == 'VIC']
    return conditions


def feature_replacement(X):
    ''' replace missing values based on specific data type of a column '''
    for col in X.columns:
        if X[col].dtype.name == 'category':
            mode = X[col].mode().iloc[0]
            X[col] = X[col].fillna(mode)
        elif X[col].dtype.name == 'datetime64[ns]':
            pseudo_date = pd.Timestamp.max
            X[col] = X[col].fillna(pseudo_date)
        elif X[col].dtype.name == 'float64':
            mean = X[col].mean()
            X[col] = X[col].fillna(mean)
        else:
            X[col].dtype.name == 'int64'
            median = X[col].median()
            X[col] = X[col].fillna(median)


def convert_DOB_to_age(df, var_name):
    ''' calculate age from DOB '''
    df[var_name] = np.where(df[var_name] > datetime.now(),
                            df[var_name] - timedelta(365*100),
                            df[var_name])
    df['age'] = datetime.now().year - df[var_name].apply(lambda x:x.year)
    

def create_unique_dataframe(df, vars_list):
    ''' subset and create unique facts & dimensions'''
    df_unique = df[vars_list]
    df_unique = df_unique.drop_duplicates()
    return df_unique


def write_as_csv(df, filename, filepath):
    ''' save the dataframe as .csv file in specific filepath '''
    df.to_csv(os.path.join(filepath, filename), index=False)
    

# --- 2. Load the data --- #
# define input CSVs:
if __name__ == '__main__':
    file_name = 'kpmg_dataset.xlsx'

# load dataframe by each sheet
df_trans = load_file_excel(file_name, 'Transactions')
df_new_cust = load_file_excel(file_name, 'NewCustomerList')
df_cust_demo = load_file_excel(file_name, 'CustomerDemographic')
df_cust_add = load_file_excel(file_name, 'CustomerAddress')

# check data types on dataframes
df_trans.dtypes
df_new_cust.dtypes
df_cust_demo.dtypes
df_cust_add.dtypes

# --- check duplicates ---
df_cust_add_clean = drop_duplicates_data(df_cust_add,'customer_id')
df_cust_demo_clean = drop_duplicates_data(df_cust_demo,'customer_id')
df_new_cust_clean = drop_duplicates_data(df_new_cust,'first_name')
df_trans_clean = drop_duplicates_data(df_trans,'customer_id')

del(df_cust_add, df_cust_demo, df_new_cust, df_trans)

# --- check missing values ---
df_missing_cust_add = eda_missing(df_cust_add_clean)
df_missing_cust_demo = eda_missing(df_cust_demo_clean)
df_missing_new_cust = eda_missing(df_new_cust_clean)
df_missing_trans = eda_missing(df_trans_clean)

# print missing % dataframes
print(df_missing_cust_add)

print(df_missing_cust_demo)

print(df_missing_new_cust)

print(df_missing_trans)

del(df_missing_cust_add, df_missing_cust_demo, df_missing_new_cust, df_missing_trans)

# --- 3. ETL: Join Data --- #
# prepare a train set: join customer_demographic with customer address
df_train = join_data(df_cust_demo_clean, df_cust_add_clean, 'left', 'customer_id')

# joins df_train with df_trans_clean
df_train = join_data(df_trans_clean, df_train, 'left', 'customer_id')

# prepare a test set: join new_customer_list with customer address
df_test = join_data(df_new_cust_clean, df_cust_add_clean, 'left', ['postcode','address'])

# joins df_test with df_trans_clean
df_test = join_data(df_test, df_trans_clean, 'left', 'customer_id')

# Subset dataframe based on list of columns
list_trans = list(df_trans_clean.columns)
list_address = list(df_cust_add_clean.columns)
list_new_cust = list(df_new_cust_clean.columns)
list_cust_demo = list(df_cust_demo_clean.columns)

# Specify unnamed columns
list_drop = ['Unnamed: 16', 'Unnamed: 19', 'Unnamed: 18', 'Unnamed: 17', 'Unnamed: 20']

# Clean the list for new customers
list_new_cust = [item for item in list_new_cust if item not in list_drop]

del(df_trans_clean, df_cust_add_clean, df_new_cust_clean, df_cust_demo_clean)

# --- 4. ETL: Data Cleaning --- #
############
# test set #
############
# remove duplicated columns: 'state_y','country_y'
df_test = drop_data(df_test, ['state_y','country_y','property_valuation_y'])

# drop columns: unnamed
df_test = drop_data(df_test, ['Unnamed: 16', 'Unnamed: 17', 'Unnamed: 18',
                        'Unnamed: 19', 'Unnamed: 20'])

# rename columns: 'state_x', 'country_x'
df_test.rename(columns={'state_x':'state','country_x':'country',
                        'property_valuation_x':'property_valuation'}, inplace=True)

# add the empty column(s): default
df_test = df_test.assign(default='')

# create a column list
col_list_1 = df_test.columns.tolist()

#############
# train set #
#############
# add the empty column(s): Rank, Value
df_train = df_train.assign(Rank='')
df_train = df_train.assign(Value='')

# create a column list
col_list_2 = df_train.columns.tolist()

# compare the lists
list_comparator(col_list_2, col_list_1)

# re-arrange the columns using col_list
df_train = df_train[col_list_1]
del(col_list_1, col_list_2)

# --- ETL: Data Conversion ---
# convert data type: object to category
convert_dt_as_cat(df_train)
convert_dt_as_cat(df_test)

# --- ETL: Feature Encoding ---
# gender: hard-coding
choices = ['Female', 'Female', 'Female', 'Unknown', 'Male', 'Male']

conditions_train = gender_conditions(df_train)
conditions_test = gender_conditions(df_test)

# apply on both train and test sets
df_train['gender'] = np.select(conditions_train, choices, default=None)
df_test['gender'] = np.select(conditions_test, choices, default=None)

# deceased_indicator: hard-coding
choices = ['Yes', 'No']

conditions_train = deceased_conditions(df_train)
conditions_test = deceased_conditions(df_test)

# apply on both train and test sets
df_train['deceased_indicator'] = np.select(conditions_train, choices, default=None)
df_test['deceased_indicator'] = np.select(conditions_test, choices, default=None)

# online_order: hard-coding
choices = ['Yes', 'No']

conditions_train = order_conditions(df_train)
conditions_test = order_conditions(df_test)

# apply on both train and test sets
df_train['online_order'] = np.select(conditions_train, choices, default=None)
df_test['online_order'] = np.select(conditions_test, choices, default=None)

# state: hard-coding
choices = ['New South Wales', 'New South Wales', 'Victoria', 'Queensland', 'Victoria']

conditions_train = state_conditions(df_train)
conditions_test = state_conditions(df_test)

# apply on both train and test sets
df_train['state'] = np.select(conditions_train, choices, default=None)
df_test['state'] = np.select(conditions_test, choices, default=None)

del(conditions_train, conditions_test)

# --- ETL: Feature Engineering ---
# calculate age from DOB #
convert_DOB_to_age(df_train, 'DOB')
convert_DOB_to_age(df_test, 'DOB')
    
# calculate age group from age
df_train['age'].describe()
df_test['age'].describe()

# remove any observations where age is greater than 90
df_train = df_train[df_train.age < 90]

# set up age group bins and labels
bins_age = [10, 19, 29, 39, 49, 59, 69, 79, 89]
labels_age = ['10s', '20s', '30s', '40s', '50s', '60s', '70s', '80s']

# calculate age group by decades
df_train['age_group'] = pd.cut(x=df_train['age'], bins=bins_age, labels=labels_age)
df_test['age_group'] = pd.cut(x=df_test['age'], bins=bins_age, labels=labels_age)

# --- ETL: Merge the existing and new customers ---
# create a column: file_type
df_train = df_train.assign(file_type='train')
df_test = df_test.assign(file_type='test')

# merge the two dataframes 
df_cust = pd.concat([df_train, df_test], axis=0)

del(df_train, df_test)

# convert data type
convert_int_to_datetime(df_cust, 'product_first_sold_date')
convert_dt_as_cat(df_cust)

# --- ETL: Feature Imputation ---
# check missing % 
eda_missing(df_cust)

# replace missing values 
feature_replacement(df_cust)

# --- Create Unique Facts and Dimensions ---
df_cust_existing = df_cust[df_cust.file_type=='train']
df_cust_existing = create_unique_dataframe(df_cust_existing, list_cust_demo + ['age', 'age_group'])

df_cust_new = df_cust[df_cust.file_type=='test']
df_cust_new = create_unique_dataframe(df_cust_new, list_new_cust  + ['age', 'age_group'])

df_cust_address = df_cust[df_cust.file_type=='train']
df_cust_address = create_unique_dataframe(df_cust_address, list_address)

df_transactions = df_cust[df_cust.file_type=='train']
df_transactions = create_unique_dataframe(df_transactions, list_trans)

df_cust_existing.info()

df_cust_new.info()

df_cust_address.info()

df_transactions.info()

# --- save the results for data quality inspection --- 
# specify the filepath
output_path = r'D:/Virtual_Internships/1_KPMG_Data Analytics/Output'

# save the prediction results
write_as_csv(df_cust_existing, 'kpmg_customers_existing.csv', output_path)
write_as_csv(df_cust_new, 'kpmg_customers_new.csv', output_path)
write_as_csv(df_cust_address, 'kpmg_customer_address.csv', output_path)
write_as_csv(df_transactions, 'kpmg_customer_trans.csv', output_path)