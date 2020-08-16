# Import and set directory:
import os
os.getcwd()

# Set path:
path = 'D:/TY Workspace/3. Data Science Portfolio/1. Completed Portfolios/P5_Walmart_Weekly_Prediction/2_Development'
os.chdir(path)

# Import all libraries #
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
import statsmodels.api as sm
import scipy.stats as sp
import sklearn.metrics as skm
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.utils import shuffle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures 
from sklearn.model_selection import KFold, train_test_split
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

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


def create_list_vars(df, dt_type):
    '''create a list of variables '''
    list_vars = list(df.select_dtypes(include=dt_type))
    return list_vars


def convert_dt_as_custom(df, var_name, dt_type):
    '''convert datatype on selected variables'''
    df[var_name] = df[var_name].astype(dt_type)
    return df[var_name]


def convert_dt_as_category(df):
    '''convert datatype from object to category'''
    for col in df.columns:
        if df[col].dtype.name == 'object':
            df[col] = df[col].astype('category')            

        
def drop_column(df, var_name):
    ''' drop a column on dataframe '''
    df = df.drop(var_name, axis=1)
    return df


def bind_data(df_train, df_test):
    ''' bind train and test dataframes as one '''
    df_concat = pd.concat([df_train, df_test], axis=0)
    return df_concat


def clean_data(raw_df):
    '''remove rows that contain invalid data or duplicate IDs'''
    clean_df = raw_df.drop_duplicates(subset='row_id')
#    clean_df = clean_df[clean_df['health__homicides_per_100k'] > 0]
    return clean_df    


def join_data(df1, df2, join_type, key=None,
              left_index=None, right_index=None):
    '''merge the dataframes by a key'''
    df_join = pd.merge(df1, df2, how=join_type, on=key,
                       left_index=False, right_index=False)
    return df_join


def avg_groupby_data(cleaned_df, num_var, cat_var, avg_var_name):
    '''groupby categorical var to calculate an average numerical feature'''
    avg_groupby_cat_val = cleaned_df.groupby(cat_var)[num_var].mean().sort_values(ascending=False)
    avg_groupby_cat_df = pd.DataFrame({cat_var:list(cleaned_df[cat_var].unique()),
                                      avg_var_name:avg_groupby_cat_val})
    avg_groupby_cat_df.reset_index(drop=True, inplace=True)
    return avg_groupby_cat_df


def encode_categorical_feature(df, var_name, map_name):
    '''encode categorical features into mapping values'''
    df[var_name] = df[var_name].map(map_name)
    return df[var_name]


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
    return df_stat_cat


def EDA_inspect_data_quality(df, col_name):
    '''compute indexes for rows with negative values on a selected feature'''
    idx_list = list(df.loc[df[col_name]<0].index)
    return idx_list


def EDA_feature_importance_plot(model, X, y):
    '''plots the feature importance plot on trained model'''
    model = model
    model.fit(X, y)
    feat_labels = X.columns
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.bar(range(X.shape[1]), importances[indices], align='center')
    plt.xticks(range(X.shape[1]), feat_labels[indices], rotation=90, fontsize=7)
    plt.xlim(-1, X.shape[1])


def apply_binning(df, new_var, old_var, bins, labels):
    '''apply binning on a selected variable'''
    df[new_var] = pd.cut(df[old_var], bins=bins,
                          labels=labels, include_lowest=True)
    return df[new_var]

            
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


def feature_scaling(X):
    '''Feature scaled data based on standardization'''
    sc_X = StandardScaler()
    X_std = sc_X.fit_transform(X)
    return X_std


def feature_selection(df, model, feat_name, Est_Coef, file_name):
    '''Creates L1 feature selected dataframe'''
    df_L1_select = pd.DataFrame({feat_name:df.columns, Est_Coef:model.coef_})[[feat_name, Est_Coef]]
    df_L1_select = df_L1_select.sort_values(by=Est_Coef, ascending=False)
    df_L1_select.to_excel(file_name)
    return df_L1_select


def format_dummy_vars(df, var_name):
    '''format dummy variables in recognized names'''
    df[var_name] = var_name + '_' + df[var_name].map(str)


def get_dummies(df, var_name):
    '''Create a list of dummy vars'''
    dummies = pd.get_dummies(df[var_name], drop_first=True)
    return dummies    

    
#def one_hot_encode_feature(df, cat_vars=None, num_vars=None):
#    '''performs one-hot encoding on all categorical variables and
#       combine results with numerical variables '''
#    cat_df = pd.get_dummies(df[cat_vars], drop_first=True)
#    num_df = df[num_vars].apply(pd.to_numeric)
#    return pd.concat([cat_df, num_df], axis=1)


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
    plt.xlabel('Predicted weekly sales')
    plt.ylabel('Actual weekly sales')
    plt.legend(loc='upper left')


def plot_reg_residual(y_train_pred, y_train_act):
    '''Plot a scatter plot to visualize residual from predicted vs. actual label'''
    plt.scatter(y_train_pred, (y_train_pred - y_train_act), c='blue',
                edgecolor='w', marker='o', label='train set')
    plt.hlines(y=0, xmin=0, xmax=max(y_train_act), colors='k', lw=3)
    plt.xlim([0, max(y_train_act)])
    plt.xlabel('Predicted weekly sales')
    plt.ylabel('Residual weekly sales')
    plt.legend(loc='upper left')    


# --- 2. Load the data --- #
# define input CSVs:
if __name__ == '__main__':
    test_data = 'sales_test.csv'
    train_data = 'sales_train.csv'
    store_data = 'sales_stores.csv'
    feature_data = 'sales_features.csv'

# Load data
df_test = load_file(test_data)
df_train = load_file(train_data)
df_store = load_file(store_data)
df_feature = load_file(feature_data)

# Create a column file_type:
# distinguish the train vs. test files
df_train['File_Type'] = 'Train'
df_test['File_Type'] = 'Test'

# re-order columns on df_train:
list_order = ['Store','Dept','Date','IsHoliday','File_Type','Weekly_Sales']
df_train = df_train[list_order]

# concatenate df_train and df_test:
df_walmart_sales = bind_data(df_train, df_test)
# delete dataframes:
del(df_train, df_test)

# --- 3. ETL - metadata format --- #
# left joins on feature and store data:
list_key_1 = 'Store'
df_dimension = join_data(df_feature, df_store, 'left', list_key_1)

# left joins on df_dimension and df_walmart_sales:
list_key_2 = ['Store', 'Date', 'IsHoliday']
df_staging = join_data(df_walmart_sales, df_dimension, 'left', list_key_2)
# delete keys and dataframes:
del(df_walmart_sales, df_dimension, df_feature, df_store)
del(list_key_1, list_key_2)

# --- 4. ETL - merging/subsetting data --- #
# define variables:
var_label = 'Weekly_Sales'
var_id_1 = 'Store'
var_id_2 = 'Dept'
var_id_3 = 'Date'
list_id_vars = [var_id_1, var_id_2, var_id_3]

vars_num_disc = create_list_vars(df_staging, 'int64') 
vars_num_cont = create_list_vars(df_staging, 'float64')

# concatenate the two lists:
vars_num = vars_num_disc + vars_num_cont

list_unwanted = {'Store','Dept'}

vars_num = [item for item in vars_num if item not in list_unwanted]

del(vars_num_disc, vars_num_cont)

# check data types on dataframes:
df_staging.info()

# create a dataframe with label:
df_label = df_staging[['Store','Dept','Date','Weekly_Sales']]

# drop a specified column:
df_staging = drop_column(df_staging, var_label)

# merge on a walmart sales and label:
df_staging_raw = join_data(df_staging, df_label, 'inner', list_id_vars)
del(df_staging)

# --- 5. ETL - cleaning data --- #
## clean data:
#df_train_clean = shuffle(clean_data(df_train_raw)).reset_index(drop=True)
#df_test_clean = shuffle(clean_data(df_test)).reset_index(drop=True)
#del(df_test, df_train, df_train_raw)
#
## create a dataframe: test set 'id'
#df_test_id = df_test_clean['row_id']

# --- 6. Feature Encoding --- #
# convert temperature "Fahrenheit" to "Celcius":
df_staging_raw['Temperature'] = (df_staging_raw['Temperature'] - 32) * 5/9

# format date as follow:
date_format = '%Y-%m-%d'
df_staging_raw['Date'] = pd.to_datetime(df_staging_raw['Date'])

# --- 7. Feature Imputation --- #
# Compute missing value %: before replacement
df_missing_pre = EDA_missing_data(df_staging_raw)
df_missing_pre

# feature imputation:
feature_replacement(df_staging_raw)

# Compute missing value %: after replacement
df_missing_post = EDA_missing_data(df_staging_raw)
df_missing_post

del(df_missing_pre, df_missing_post)

# --- 8. Feature Engineering --- #
# assign a feature: binary flag
#df_staging_raw = df_staging_raw.assign(md1_presence = df_staging_raw.MarkDown1.notnull())
#df_staging_raw = df_staging_raw.assign(md2_presence = df_staging_raw.MarkDown2.notnull())
#df_staging_raw = df_staging_raw.assign(md3_presence = df_staging_raw.MarkDown3.notnull())
#df_staging_raw = df_staging_raw.assign(md4_presence = df_staging_raw.MarkDown4.notnull())
#df_staging_raw = df_staging_raw.assign(md5_presence = df_staging_raw.MarkDown5.notnull())

## average numeric by categorical variables:
#df_avg_poverty_by_econ = avg_groupby_data(df_train_clean, 'poverty_rate', 'econ__economic_typology', 'avg_poverty_by_econ_type')
## perform left joins on avg grouped dataframes:
#df_train_clean  = join_data(df_train_clean, df_avg_poverty_by_econ, 'left', key='econ__economic_typology')
#df_test_clean = join_data(df_test_clean, df_avg_poverty_by_econ, 'left', key='econ__economic_typology')
#del(df_avg_poverty_by_econ)

# --- 9. Exploratory Data Analysis --- #
# perform numerical stat:
df_stat_num = EDA_summary_stat_num(df_staging_raw[vars_num])
df_stat_num

# convert data type: store and dept as object
convert_dt_as_custom(df_staging_raw, 'Store', 'object')
convert_dt_as_custom(df_staging_raw, 'Dept', 'object')

# convert data type as category:
convert_dt_as_category(df_staging_raw)

# create a list: categorical variables
vars_cat = create_list_vars(df_staging_raw, 'category')

# perform summary statistics: categorical
df_stat_cat = EDA_summary_stat_cat(df_staging_raw[vars_cat])
df_stat_cat

# --- 10. Prepare Training Data --- #
# Create a dummy variable: date
df_staging_raw['Date_String'] = [datetime.strptime(date, '%Y-%m-%d').date() 
                            for date in df_staging_raw['Date'].astype(str).values.tolist()]
df_staging_raw['Month'] = [date.month for date in df_staging_raw['Date_String']]

# Create a list of dummy variables: Special Dates - Black Friday & Christmas Eve
df_staging_raw['Black_Friday'] = np.where((df_staging_raw['Date_String']==datetime(2010, 11, 26).date()) | 
        (df_staging_raw['Date_String']==datetime(2011, 11, 25).date()), 'yes', 'no')
df_staging_raw['Christmas_Eves'] = np.where((df_staging_raw['Date_String']==datetime(2010, 12, 23).date()) | 
        (df_staging_raw['Date_String']==datetime(2010, 12, 24).date()) | 
        (df_staging_raw['Date_String']==datetime(2011, 12, 23).date()) | 
        (df_staging_raw['Date_String']==datetime(2011, 12, 24).date()), 'yes', 'no')

# Re-format dummy variables: Month, IsHoliday, etc.
format_dummy_vars(df_staging_raw, 'Month')
format_dummy_vars(df_staging_raw, 'IsHoliday')
format_dummy_vars(df_staging_raw, 'Black_Friday')
format_dummy_vars(df_staging_raw, 'Christmas_Eves')

# Create a list of dummy variables:
#dummy_months = get_dummies(df_staging_raw, 'Month')
dummy_holidays = get_dummies(df_staging_raw, 'IsHoliday')
dummy_black_friday = get_dummies(df_staging_raw, 'Black_Friday')
dummy_christmas_eve = get_dummies(df_staging_raw, 'Christmas_Eves')

# Concatenate dataframe:
df_clean = pd.concat([df_staging_raw, dummy_christmas_eve, 
                      dummy_black_friday, dummy_holidays], axis=1)

del(df_staging_raw)
del(dummy_holidays, dummy_black_friday, dummy_christmas_eve)

# --- 11. Feature Engineering --- #
# create median sales by store and department on a train set:
df_median_sales = pd.DataFrame({'Median_Sales':df_clean.loc[df_clean['File_Type']=='Train'].groupby(
                  by=['Type','Dept','Store','Month','IsHoliday'])['Weekly_Sales'].median()}).reset_index()

# check dataframe:
df_median_sales.head()

# perform left joins on df_clean and df_median_sales:
list_key_3 = ['Type','Dept','Store','Month','IsHoliday']
df_clean = join_data(df_clean, df_median_sales, 'left', list_key_3)
del(df_median_sales)

# imputed Median Sales:
feature_replacement(df_clean)

# create a feature: Date_Lagged_1wk
df_clean['Date_1wk_Lagged'] = df_clean['Date_String'] - timedelta(days=7)
df_clean.head()

# Create a sorted dataframe: 1wk lagged weekly sales computation
df_clean = df_clean.sort_values(['Store', 'Dept', 'Date_String'], ascending=[True,True,True])
df_clean = df_clean.reset_index(drop=True)

# Create Lagged_1wk_Sales, Lagged_1wk_Available
df_clean['Lagged_1wk_Sales'] = np.nan
df_clean['Lagged_1wk_Available'] = np.nan

# specify row's: last and length
row_last = df_clean.loc[0]
row_len = df_clean.shape[0]

# for looping on each row:
for index, row in df_clean.iterrows():
    lag_date = row['Date_1wk_Lagged']
    # Check if it matches the last week sales to the compared date
    if((row_last['Date_String'] == lag_date) & (row_last['Weekly_Sales']>0)):
        df_clean.set_value(index, 'Lagged_1wk_Sales', row_last['Weekly_Sales'])
        df_clean.set_value(index, 'Lagged_1wk_Available', 1)
    else: #last week sales doesn't exist then replace with median sales
        df_clean.set_value(index, 'Lagged_1wk_Sales', row['Median_Sales'])
        df_clean.set_value(index, 'Lagged_1wk_Available', 0)
    row_last = row # compute last row for computational speed
    if(index%int(row_len/10)==0): # shows progress by printing every 10% interval
        print(str(int(index*100/row_len))+'% computed')

# checked dataframe:
df_clean[['Dept','Store','Date_String','Lagged_1wk_Sales','Weekly_Sales','Median_Sales']].head()        

# compute Sales Variance: Median_Sales - Lagged_1wk_Sales
df_clean['Sales_Var_Median_Lagged'] = df_clean['Median_Sales'] - df_clean['Lagged_1wk_Sales']

# compute Sales Variance: Median_Sales - Weekly_Sales
df_clean['Sales_Var_Median_Weekly'] = df_clean['Median_Sales'] - df_clean['Weekly_Sales']

# --- 12. Feature Scaling --- # 
# Normalized features based on standardization
#df_clean['Temperature'] = feature_scaler(df_clean, 'Temperature')

# --- 13. split data into train and test set ---
# split the data for train and test sets:
df_final_train, df_final_test = split_dataframe(df_clean)
del(df_clean)

# create a label:
label_df = df_final_train['Sales_Var_Median_Weekly']

# feature selection:
df_final_train.info()

# selected list of features:
list_feat = list(df_final_train.columns)

# create a list: object, category and datetime
list_obj = create_list_vars(df_final_train, 'O')
list_cat = create_list_vars(df_final_train, 'category')
list_dt_time = create_list_vars(df_final_train, 'datetime')
list_target = ['Weekly_Sales', 'Median_Sales', 'Sales_Var_Median_Weekly']

# concatenate a list of features to be removed:    
list_remove = list_obj + list_cat + list_dt_time + list_target

list_feat = [item for item in list_feat if item not in list_remove]
del(list_obj, list_cat, list_dt_time, list_target, list_remove)

# split data into train and validation sets:
X_train, X_val, y_train, y_val = train_test_split(df_final_train[list_feat], label_df, 
                                                  test_size=1/4, random_state=0)

# check the split:
y_train.count()
y_val.count()

# Drop first dummy variable on each nominal feature to avoid dummy variable trap:
#df_feature_enc = one_hot_encode_feature(df_train_clean, cat_vars=vars_cat, num_vars=vars_num)
#df_test_enc = one_hot_encode_feature(df_test_clean, cat_vars=vars_cat, num_vars=vars_num)


# --- 14. Baseline model --- # 
# Baseline Model LR: MAE = 1566.47, CV_STD_MAE = 9.40    
# create a baseline model:
baseline = LinearRegression()
baseline_cv = cross_val_score(baseline, X_train, y_train, 
                                  cv=5, scoring='neg_mean_absolute_error')
baseline_mae = -1 * baseline_cv
baseline_mean_mae = np.mean(baseline_mae)
print('Baseline MAE: ' + str(baseline_mean_mae))

############################
# Part 3 - DEVELOP PHASE ###
############################
# --- 15. Create models --- # 
# initialize model list and dicts
models = []
mean_mae = {}
cv_std = {}
res = {}

# define common model parameters: num processors and shared model parameters
n_proc = 1
verbose_lv = 5

# create and tune the models that you brainstormed during part 2
###############################################################################        
# Hyper-parameters tuning: RandomForest
rf = RandomForestRegressor()   
n_iter = 10
param_dist_rf = {'n_estimators':sp.randint(10,50), 
                  'max_depth':sp.randint(1,10),
                  'min_samples_split':sp.randint(10,60)}

random_search_rf = model_tuning_param(rf, X_train, y_train, param_dist_rf, n_iter)    
    
# print the best model parameters: RandomForest    
param_1 = 'n_estimators' 
param_2 = 'max_depth'
param_3 = 'min_samples_split'
    
print_best_params(random_search_rf, param_1, param_2, param_3)
###############################################################################    

## Hyper-parameters tuning: GradientBoosting
#gbr = GradientBoostingRegressor()
#n_iter = 10
#param_dist_gbr = {'n_estimators':sp.randint(10,40), 
#                  'max_depth':sp.randint(1,20),
#                  'loss':['ls']}
#
#random_search_gbr = model_tuning_param(gbr, X_train, y_train, param_dist_gbr, n_iter)    
#    
## print the best model parameters: GradientBoosting    
#param_1 = 'n_estimators' 
#param_2 = 'max_depth'
#param_3 = 'loss'
#    
#print_best_params(random_search_gbr, param_1, param_2, param_3)        
###############################################################################    

# Hyper-parameters tuning: XGBoost
xgb = XGBRegressor()   
n_iter = 10
param_dist_xgb = {'n_estimators':sp.randint(10,40), 
                  'max_depth':sp.randint(1,20),
                  'learning_rate':np.random.uniform(0,1,10)}

random_search_xgb = model_tuning_param(xgb, X_train, y_train, param_dist_xgb, n_iter)    
    
# print the best model parameters: XGBRegressor    
param_1 = 'n_estimators' 
param_2 = 'max_depth'
param_3 = 'learning_rate'
    
print_best_params(random_search_xgb, param_1, param_2, param_3)   

 # --- 16. Cross-validate models --- # 
# do 5-fold cross validation on models and measure MSE
# Model List to train: Order of Model Complexity
lr_L1 = Lasso(alpha=0.01)
lr_L2 = Ridge(alpha=0.01)
rf = RandomForestRegressor(n_estimators=10, n_jobs=n_proc, max_depth=9,
                               min_samples_split=31, verbose=verbose_lv)   
#gbr = GradientBoostingRegressor(n_estimators=34, max_depth=8, loss='ls', verbose=verbose_lv)
xgb = XGBRegressor(n_estimators=36, max_depth=16, learning_rate=0.47767430119511955) 

# a list of regressors:
#models.extend([lr_L1, lr_L2, rf, gbr, xgb])    
models.extend([lr_L1, lr_L2, rf, xgb])    

# cross-validate models, using MSE to evaluate and print the summaries
print("begin cross-validation")
for model in models:
    model_train(model, X_train, y_train, n_proc, mean_mae, cv_std)
    model_summary(model, mean_mae, cv_std)

# --- 17. Select the best model with lowest MAE for your prediction model --- #
# Best Model XGB: MAE = 1297.21, CV_STD_MAE = 6.743    
model = min(mean_mae, key=mean_mae.get)
print('\nBest model with the lowest MAE:')
print(model)

# --- 18. Model Evaluation: Scatter Plot --- #
# Prepare predicted and original weekly sales
y_val_act = y_val.copy()

# re-train a model with best model:
model.fit(X_train, y_train)

# make predictions on "variance sales: median - weekly 
y_train_pred_var_med_weekly = model.predict(X_val)
# make actuals on "variance sales: median - weekly 
y_val_act = y_val_act.to_frame()
y_val_act['Sales_Var_Med_Weekly_Pred'] = y_train_pred_var_med_weekly

# create a final validation dataframe:
df_val = pd.merge(df_final_train, y_val_act[['Sales_Var_Med_Weekly_Pred']], how='left',
                   left_index=True, right_index=True, suffixes=['_Actual','_Pred'])

# remove rows where Sales_Var_Med_Weekly_Pred is NULL:
df_val = df_val[~pd.isnull(df_val['Sales_Var_Med_Weekly_Pred'])]
df_val.head()

# derive the weekly sales prections: median - (median - weekly_Pred)
df_val['Weekly_Sales_Pred'] = df_val['Median_Sales'] - df_val['Sales_Var_Med_Weekly_Pred']
        
# Plot a comparison scatter plot: predicted vs. actual
plt.figure(figsize=(14,7))
plt.subplots_adjust(wspace=0.2)
plt.subplot(1,2,1)
plot_reg_compare(df_val['Weekly_Sales_Pred'], df_val['Weekly_Sales'])
plt.title('Weekly sales of predicted vs. actual: best trained model')

# Plot a residual scatter plot: predicted vs. actual
plt.subplot(1,2,2)
plot_reg_residual(df_val['Weekly_Sales_Pred'], df_val['Weekly_Sales'])
plt.title('Weekly sales of predicted vs. residual: best trained model')
plt.show()

###########################
# Part 4 - DEPLOY PHASE ###
###########################
# --- 19. Automate the model pipeline --- #
# make predictions on a test set
df_pred_test = model.predict(df_final_test[list_feat])
df_pred_test = pd.DataFrame(df_pred_test)
df_pred_test.columns = ['Sales_Var_Med_Weekly_Pred']

# make predictions dataframe:
results = pd.concat([df_final_test.reset_index(drop=True), df_pred_test], axis=1)
results['Weekly_Sales_Pred'] = results['Median_Sales']-results['Sales_Var_Med_Weekly_Pred']
results = results[['Store', 'Dept', 'Date_String', 'Weekly_Sales_Pred']]

# create a final predicted dataframe:
results = gen_predictions_file(results, 'Store', 'Dept', 'Date_String')
results.head()

# --- 20. Deploy the solution --- #
#store feature importances
if hasattr(model, 'feature_importances_'):
    importances = model.feature_importances_
else:
#linear models don't have feature_importances
    importances = [0]*len(df_final_train[list_feat].columns)
    
feature_importances = pd.DataFrame({'feature':df_final_train[list_feat].columns, 
                                        'importance':importances})
feature_importances.sort_values(by='importance', ascending=False,
                                    inplace=True)
    
#set index to 'feature'
feature_importances.set_index('feature', inplace=True, drop=True)
    
#create a plot
feature_importances[0:20].plot.bar(align='center')
plt.xticks(rotation=90, fontsize=9)
plt.title('feature importance plot: best trained model')
plt.show()
    
#Save model results as .csv file:
model_results(model, mean_mae[model], results, feature_importances)