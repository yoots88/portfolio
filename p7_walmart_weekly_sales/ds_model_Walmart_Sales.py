# Import and set directory:
import os
os.getcwd()

# Set path:
#path = 'D:/DS Portfolio/P_Walmart_Sales_Time_Series/A. Coding'
path = 'D:/TY Workspace/3. Data Science Portfolio/P_Walmart_Sales_Time_Series/A. Coding'
os.chdir(path)

# Import all libraries #
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
from sklearn.preprocessing import Imputer, StandardScaler, PolynomialFeatures 
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

#############################
# Part 2 - DISCOVER PHASE ###
#############################

# --- 2. Write Out List of Functions --- #
def load_file(file):
    '''load csv file as a dataframe'''
    df = pd.read_csv(file)
    return df

def drop_column_by_index(df, var_list):
    '''drop a column by index'''
    df = df.drop(var_list, axis=1)
    return df

def join_data(df_1, df_2, join_type, left_keys, right_keys,
              left_index=None, right_index=None):
    '''merge the dataframe(s) based on the key'''
    df_join = pd.merge(df_1, df_2, how=join_type, left_on=left_keys,
                       right_on=right_keys, left_index=False, right_index=False)
    return df_join

def clean_data(df, key):
    '''drop any duplicates based on a specified column'''
    clean_df = df.drop_duplicates(subset=key)
    return clean_df

def convert_dt_as_category(df):
    '''convert data type from object to category'''
    for var_name in df.columns:
        if df[var_name].dtype.name == 'object':
            df[var_name] = df[var_name].astype('category')
            
def convert_dt_as_object(df):
    '''convert data type from category to object'''
    for var_name in df.columns:
        if df[var_name].dtype.name == 'category':
            df[var_name] = df[var_name].astype('object')
            
def EDA_missing_data(df):
    '''computes missing % on train data'''
    missing_df = pd.DataFrame(df.isnull().sum())
    missing_df.columns = ['count']
    missing_df = missing_df.sort_values(by='count', ascending=False)
    missing_df['pct'] = missing_df['count']/len(df)
    return missing_df

def EDA_summary_stat_num(df):
    '''shows the summary statistics on numerical var'''
    df_stat_num = df.describe().T
    df_stat_num = df_stat_num[['count', 'min', 'max', 'mean', 'std', '25%', '50%', '75%']]
    df_stat_num = df_stat_num.sort_values(by='count', ascending=True)
    df_stat_num = pd.DataFrame(df_stat_num)
    return df_stat_num

def EDA_summary_stat_cat(df):
    '''shows the summary statistics on categorical var'''
    df_stat_cat = df.describe(include='O').T
    df_stat_cat = pd.DataFrame(df_stat_cat)
    return df_stat_cat

def EDA_plot_corr_matrix(df, features, label):
    '''plot the correlation matrix'''
    corr = df[features+label].corr()
    # Create a mask:
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    plt.figure(figsize=(12,10))
    sns.heatmap(corr, cmap=sns.diverging_palette(220, 10, as_cmap=True),
                annot=True, fmt=".2f", mask=mask)
    plt.xticks(rotation=90)
    plt.show()

def EDA_plot_scatter(df, 
                     var1, lab1, c1,
                     var2, lab2, c2,
                     factor=None):
    '''plot a scatter plot of 2 by 1 figure'''
    plt.figure(figsize=(8,8))
    plt.subplots_adjust(hspace=0.4, wspace=0.9)
    plt.subplot(2,1,1)
    plt.scatter(df[var1]*factor, df['Weekly_Sales'], color=c1)
    plt.title('Scatterplot between '+ lab1 + ' and Weekly Sales')
    plt.xlabel(lab1)
    plt.ylabel('Weekly_Sales')
    
    plt.subplot(2,1,2)
    plt.scatter(df[var2]*factor, df['Weekly_Sales'], color=c2)
    plt.title('Scatterplot between '+ lab2 + ' and Weekly Sales')
    plt.xlabel(lab2)
    plt.ylabel('Weekly_Sales')

def missing_val_imputer(df):
    '''impute missing values based on specific data type and a column'''
    for column in df.columns:
        if df[column].dtype.name == 'object':
            mode = df[column].mode()
            df[column] = df[column].fillna(str(mode))
        elif df[column].dtype.name == 'float64':
            mean = df[column].mean()
            df[column] = df[column].fillna(mean)
        elif df[column].dtype.name == 'datetime64[ns]':
            pseudo_date = pd.Timestamp.max
            df[column] = df[column].fillna(pseudo_date)
        else:
            md_cols = df.columns[df.columns.str.contains(pat = 'MarkDown')]
            df[md_cols] = df[md_cols].fillna(0)

def split_dataframe(df):
    '''Split data into train and test sets'''
    df_final_train, df_final_test = df.loc[df['File_Type']=='Train'], df.loc[df['File_Type']=='Test']
    return df_final_train, df_final_test

def format_dummy_vars(df, var_name):
    '''Format dummy variables in recognized names'''
    df[var_name] = var_name + '_' + df[var_name].map(str)

def get_dummies(df, var_name):
    '''Create a list of dummy vars'''
    dummies = pd.get_dummies(df[var_name], drop_first=True)
    return dummies

def feature_scaler(df, var_name):
    '''scale feature based on standard normalization'''
    df[var_name] = (df[var_name] - df[var_name].mean())/(df[var_name].std())
    

def model_tuning_param(model, X_train, y_train, param_dist, n_iter):
    '''Performs RandomizedSearchCV to tune model parameters'''
    random_search = RandomizedSearchCV(model, param_dist, n_iter, cv=2)
    random_search.fit(X_train, y_train)
    return random_search


def print_best_params(random_search, param_1=None, param_2=None, param_3=None):
    '''Print the best model parameters'''
    print("Best " + param_1 + ":", random_search.best_estimator_.get_params()[param_1])
    print("Best " + param_2 + ":", random_search.best_estimator_.get_params()[param_2])
    print("Best " + param_3 + ":", random_search.best_estimator_.get_params()[param_3])
    

def model_train(model, X_train, y_train, n_proc, mean_mae, cv_std):
    '''Train a model and output mean MAE and CV Std. MAE'''
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

def gen_predictions_file(df, var_1, var_2, var_3):
    '''Generate test predictions file: concatenate with 3 keys to make the id column'''
    df['id'] = df[var_1].map(str)+'_'+df[var_2].map(str)+'_'+df[var_3].map(str)
    return df[['id', 'Weekly_Sales_Pred']]


def model_results(model, mean_mae, predictions, feature_importances):
    '''Saves the model name, mean_mae, predicted sales, and feature importances'''
    with open('model.txt', 'w') as file:
        file.write(str(model))
        feature_importances.to_csv('feat_importances.csv',index=False)
        df_predictions.to_csv('final_predictions.csv',index=False)


def compute_confusion_matrix(y_act, y_pred):
    '''compute sklearn confusion matrix'''
    cm_model = skm.confusion_matrix(y_act, y_pred)
    return cm_model 


def plot_confusion_matrix(cm, classes):
    '''plot the confusion matrix of trained model'''
    fig, ax = plt.subplots(figsize=(7,7))
    cm = cm.astype('float')/cm.sum()
    
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt='.2f'
    thresh = cm.max()/2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i,j], fmt), ha='center', va='center',
                    color='white' if cm[i,j] > thresh else 'black')
    plt.xlabel('predicted label')
    plt.ylabel('true label')


def plot_reg_compare(y_train_pred, y_train_act):
    '''Plot a scatter plot to compare predicted vs actual label'''
    plt.scatter(y_train_act, y_train_pred, c='blue', 
                edgecolor='w', marker='o', label='train set')
    plt.xlabel('Actual Weekly Sales')
    plt.ylabel('Predicted Weekly Sales')
    plt.legend(loc='upper left')
    
    
def plot_reg_residual(y_train_pred, y_train_act):
    '''Plot a scatter plot to visualize residual from predicted vs. actual label'''
    plt.scatter(y_train_pred, (y_train_pred - y_train_act),  c='blue',
                edgecolor='w', marker='o', label='train set')
    plt.hlines(y=0, xmin=-100000, xmax=500000, colors='k', lw=3)
    plt.ylim([-100000,100000])
    plt.xlabel('Predicted Weekly Sales')
    plt.ylabel('Residual Weekly Sales')
    plt.legend(loc='upper left')


# --- 3. Load the data --- #
# Define input CSVs:
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

# Create a column file_type: distinguish train & test files for model training
df_train['File_Type'] = 'Train'
df_test['File_Type'] = 'Test'

# Re-order columns on df_train:
df_train = df_train[['Store','Dept','Date','IsHoliday','File_Type','Weekly_Sales']]


# Concatenate df_train and df_test as one dataframe:
df = pd.concat([df_train, df_test], axis=0)

# 1st Join: left join on the feature and store data
df_join_1 = join_data(df_feature, df_store, 'left', 'Store', 'Store')
# Final Join: left join on the train and joined dataframe
df_join_final = join_data(df, df_join_1, 'left', 
                          ['Store', 'Date', 'IsHoliday'], ['Store', 'Date', 'IsHoliday'])

del(df, df_join_1)
del(df_test, df_train, df_store, df_feature)

# Define type of variables list:
#df_join_final.select_dtypes(include='object').columns
#df_join_final.select_dtypes(include='int64').columns
#df_join_final.select_dtypes(include='float64').columns

# Define variables: 
features = ['Date', 'Type', 'Store', 'Dept', 'Size', 'Temperature', 'Fuel_Price',
            'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5',
            'CPI', 'Unemployment' ]

label_var = ['Weekly_Sales']

id_var_1 = 'Store'
id_var_2 = 'Dept'
id_var_3 = 'Date'

var_list = [id_var_1, id_var_2, id_var_3]

# Drop a column by index: poverty_rate
df_label = df_join_final[['Store', 'Dept', 'Date', 'Weekly_Sales']]
df_staging = drop_column_by_index(df_join_final, label_var)
del df_join_final

# Join the train and label:
df_raw = join_data(df_staging, df_label, 'inner', var_list, var_list)
del df_staging

# --- 4. Perform data cleaning and quality check --- #
# Compute missing value % on a dataframe:
# MarkDown 1-5 missing values more than 50%
missing_df_train = EDA_missing_data(df_raw)
missing_df_train

# --- 5. Feature encoding on numerical variables --- #
# Convert the temperature from Fahrenheit to Celsius
df_raw['Temperature'] = (df_raw['Temperature'] - 32) * 5/9

# Format Date based on the date_format
date_format = '%Y-%m-%d'
df_raw['Date'] = pd.to_datetime(df_raw['Date'])

# --- 6. Explore the data (EDA) --- # 
# Compute summary statistics
df_stat_num = EDA_summary_stat_num(df_raw)
df_stat_num

df_stat_cat = EDA_summary_stat_cat(df_raw)
df_stat_cat


# Correlation matrix plot # 
###########################    
# Compute the correlation of each feature against weekly sales: order of magnitude
EDA_plot_corr_matrix(df_raw, features, label_var)


EDA_plot_scatter(df_raw, 'Fuel_Price', 'Fuel_Price', 'blue',
                 'Size', 'Size', 'green', 1)    

# Verify the Weekly Sales > 300K:
df_raw.loc[df_raw['Weekly_Sales'] > 300000]

# Check the dates where Weekly_Sales > 240K:
# 2 days before Christmas and Black Friday (*)
df_raw.loc[df_raw['Weekly_Sales'] > 240000, "Date"].value_counts()


# --- 7. Feature engineering on categorical variables --- #
#############################################################
# Feature Engineering: indicate(s) presence of MarkDown 1-5 #
#############################################################
df_raw = df_raw.assign(md1_presence = df_raw.MarkDown1.notnull())
df_raw = df_raw.assign(md2_presence = df_raw.MarkDown2.notnull())
df_raw = df_raw.assign(md3_presence = df_raw.MarkDown3.notnull())
df_raw = df_raw.assign(md4_presence = df_raw.MarkDown4.notnull())
df_raw = df_raw.assign(md5_presence = df_raw.MarkDown5.notnull())

# --- 8. Feature imputation via univariate techniques --- #    
# make df_imputed:
df_imputed = df_raw.copy()
df_imputed.dtypes

# For all missing values on MarkDown 1-5, just replaced them with "0"
missing_val_imputer(df_imputed)

# check any missing values on imputed df:
df_imputed.isnull().sum()


# --- 9. One-hot-encode on features --- # 
# Drop first dummy variable to avoid dummy variable trap on each converted feature!
# Create a dummy variable: date
df_imputed['Date_String'] = [datetime.strptime(date, '%Y-%m-%d').date() 
                            for date in df_imputed['Date'].astype(str).values.tolist()]
df_imputed['Month'] = [date.month for date in df_imputed['Date_String']]

# Create a list of dummy variables: Special Dates - Black Friday & Christmas Eve
df_imputed['Black_Friday'] = np.where((df_imputed['Date_String']==datetime(2010, 11, 26).date()) | 
        (df_imputed['Date_String']==datetime(2011, 11, 25).date()), 'yes', 'no')
df_imputed['Christmas_Eves'] = np.where((df_imputed['Date_String']==datetime(2010, 12, 23).date()) | 
        (df_imputed['Date_String']==datetime(2010, 12, 24).date()) | 
        (df_imputed['Date_String']==datetime(2011, 12, 23).date()) | 
        (df_imputed['Date_String']==datetime(2011, 12, 24).date()), 'yes', 'no')

# Re-format dummy variables: Type, Store, Dept and IsHoliday
#format_dummy_vars(df_imputed, 'Type')
#format_dummy_vars(df_imputed, 'Store')
#format_dummy_vars(df_imputed, 'Dept')
format_dummy_vars(df_imputed, 'Month')
format_dummy_vars(df_imputed, 'IsHoliday')
format_dummy_vars(df_imputed, 'Black_Friday')
format_dummy_vars(df_imputed, 'Christmas_Eves')

# Create a list of dummy variables:
#type_dummies = get_dummies(df_imputed, 'Type')
#store_dummies = get_dummies(df_imputed, 'Store') 
#dept_dummies = get_dummies(df_imputed, 'Dept')
month_dummies = get_dummies(df_imputed, 'Month')
holiday_dummies = get_dummies(df_imputed, 'IsHoliday')
black_friday_dummies = get_dummies(df_imputed, 'Black_Friday')
christmas_eve_dummies = get_dummies(df_imputed, 'Christmas_Eves')

del df_raw

# Concatenate dataframe: use the special date dummies for now #
df_cleaned = pd.concat([df_imputed, christmas_eve_dummies, 
                              black_friday_dummies, holiday_dummies], axis=1)


# --- 10. Feature engineering on numerical variables --- #
# Create store median sales by store and department on train set only!
df_median_sales = pd.DataFrame({'Median_Sales':df_cleaned.loc[df_cleaned['File_Type']=='Train'].groupby(
                  by=['Type','Dept','Store','Month','IsHoliday'])['Weekly_Sales'].median()}).reset_index()

df_median_sales.head()

# Perform outer join on df_cleaned and df_median_sales:
df_cleaned = join_data(df_cleaned, df_median_sales, 'outer',
                           ['Type','Dept','Store','Month','IsHoliday'],
                           ['Type','Dept','Store','Month','IsHoliday'])
del(df_median_sales)

# Imputed Median_Sales for NAs:
df_cleaned['Median_Sales'].fillna(df_cleaned['Median_Sales'].loc[df_cleaned['File_Type']=='Train'].median()
                                  ,inplace=True)

# Create a lagging variable: store's 1wk lagged weekly sales
df_cleaned['Date_Lagged_1wk'] = df_cleaned['Date_String'] - timedelta(days=7)
df_cleaned.head()

# Create a sorted dataframe: 1wk lagged weekly sales computation
df_lagged_sales = df_cleaned.sort_values(['Store', 'Dept', 'Date_String'], ascending=[True,True,True])
df_lagged_sales = df_lagged_sales.reset_index(drop=True)

# Create Lagged_1wk_Sales, Lagged_1wk_Available
# Initialize features: Lagged_1wk_Sales, Lagged_1wk_Available
df_lagged_sales['Lagged_1wk_Sales'] = np.nan
df_lagged_sales['Lagged_1wk_Available'] = np.nan
row_last = df_cleaned.loc[0]
row_len = df_lagged_sales.shape[0]
for index, row in df_lagged_sales.iterrows():
    lag_date = row['Date_Lagged_1wk']
    # Check if it matches the last week sales to the compared date
    if((row_last['Date_String'] == lag_date) & (row_last['Weekly_Sales']>0)):
        df_lagged_sales.set_value(index, 'Lagged_1wk_Sales', row_last['Weekly_Sales'])
        df_lagged_sales.set_value(index, 'Lagged_1wk_Available', 1)
    else: #replace with median sales
        df_lagged_sales.set_value(index, 'Lagged_1wk_Sales', row['Median_Sales'])
        df_lagged_sales.set_value(index, 'Lagged_1wk_Available', 0)
    row_last = row # compute last row for computational speed
    if(index%int(row_len/10)==0): # shows progress by printing every 10% interval
        print(str(int(index*100/row_len))+'% computed')
        
# Checked the engineered features:
df_lagged_sales[['Dept','Store','Date_String','Lagged_1wk_Sales','Weekly_Sales','Median_Sales']].head()        

df_cleaned = df_lagged_sales.copy()
del(df_lagged_sales)

# Compute Variance_Sales: Median_Sales - Lagged_1wk_Sales
df_cleaned['Sales_Var_Median_Lagged'] = df_cleaned['Median_Sales'] - df_cleaned['Lagged_1wk_Sales']

# Checked the engineered features:
df_cleaned[['Dept','Store','Date_String','Weekly_Sales','Median_Sales','Lagged_1wk_Sales','Sales_Var_Median_Lagged']].head()

# --- 11. Feature Scaling --- # 
# Normalized features based on standardization
#df_cleaned['Temperature'] = feature_scaler(df_cleaned, 'Temperature')

# Compute Variance_Sales: Median_Sales - Weekly_Sales
df_cleaned['Sales_Var_Median_Weekly'] = df_cleaned['Median_Sales'] - df_cleaned['Weekly_Sales']


# --- 12. Feature selection --- # 
selection = ['CPI', 'Fuel_Price', 'Size', 'Temperature', 'Unemployment',
             'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5',
             'md1_presence','md2_presence', 'md3_presence', 'md4_presence', 'md5_presence', 
             'IsHoliday_True', 'Christmas_Eves_yes', 'Black_Friday_yes', 
             'Lagged_1wk_Sales', 'Lagged_1wk_Available', 'Sales_Var_Median_Lagged']

# Compute the EDA on selected features:
df_cleaned[selection].describe()
df_cleaned[selection].head()

# Save feature_df for EDA portfolio:
df_eda = df_cleaned[df_cleaned['File_Type']=='Train']
df_eda.to_csv('df_eda.csv', index=False)

# --- 13. Establish a baseline model --- # 
# Split the data for train and test sets:
df_final_train, df_final_test = split_dataframe(df_cleaned)

# Spit the data into features and label:
np.random.seed(100)
X_train, X_val, y_train, y_val = train_test_split(df_final_train[selection], df_final_train['Sales_Var_Median_Weekly'],
                                                  test_size=1/4, random_state=100)
print(X_val.shape)
print(y_val.shape)


# Establish a baseline model:
baseline = LinearRegression()
baseline_cv = cross_val_score(baseline, X_train,  y_train.ravel(),
                              cv=5, scoring='neg_mean_absolute_error')
baseline_mae = -1 * baseline_cv
baseline_mean_mae = np.mean(baseline_mae)
print('Baseline MAE: ' + str(baseline_mean_mae))

############################
# Part 3 - DEVELOP PHASE ###
############################
# --- 14. Create models --- # 
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
n_iter=10
param_dist_rf = {'n_estimators':sp.randint(10,80), 
                  'min_samples_split':sp.randint(2,10),
                  'min_samples_leaf':sp.randint(2,10)}

random_search_rf = model_tuning_param(rf, X_train, y_train.ravel(), param_dist_rf, n_iter)   

# print the best model parameters: RandomForest    
param_1 = 'n_estimators' 
param_2 = 'min_samples_split'
param_3 = 'min_samples_leaf'
    
print_best_params(random_search_rf, param_1, param_2, param_3)
###############################################################################    
# Hyper-parameters tuning: GradientBoosting
gbr = GradientBoostingRegressor()
n_iter = 10
param_dist_gbr = {'n_estimators':sp.randint(10,40), 
                  'max_depth':sp.randint(2,20),
                  'loss':['ls']}

random_search_gbr = model_tuning_param(gbr, X_train, y_train.ravel(), param_dist_gbr, n_iter)    
    
# print the best model parameters: GradientBoosting    
param_1 = 'n_estimators' 
param_2 = 'max_depth'
param_3 = 'loss'
    
print_best_params(random_search_gbr, param_1, param_2, param_3) 
###############################################################################    
# Hyper-parameters tuning: XGBoost
xgb = XGBRegressor()   
n_iter = 10
param_dist_xgb = {'n_estimators':sp.randint(10,40), 
                  'max_depth':sp.randint(2,20),
                  'learning_rate':np.random.uniform(0,1,10)}

random_search_xgb = model_tuning_param(xgb, X_train, y_train.ravel(), param_dist_xgb, n_iter)    
    
# print the best model parameters: XGBRegressor    
param_1 = 'n_estimators' 
param_2 = 'max_depth'
param_3 = 'learning_rate'
    
print_best_params(random_search_xgb, param_1, param_2, param_3)
###############################################################################    
# --- 15. Cross-validate models --- # 
#do 5-fold cross validation on models and measure MAE
# Model List to train: Order of Model Complexity
rf = RandomForestRegressor(n_estimators=61, n_jobs=n_proc, min_samples_split=2,
                               min_samples_leaf=4, verbose=verbose_lv)   
gbr = GradientBoostingRegressor(n_estimators=38, max_depth=10, loss='ls', verbose=verbose_lv)
xgb = XGBRegressor(n_estimators=24, max_depth=10, learning_rate=0.18949843940084687) 

models.extend([rf, gbr, xgb])    

# cross-validate models, using MSE to evaluate and print the summaries
print("begin cross-validation")
for model in models:
    model_train(model, X_train, y_train.ravel(), n_proc, mean_mae, cv_std)
    model_summary(model, mean_mae, cv_std)
    
# --- 16. Select the best model with lowest MAE for predictions --- #
# Best Model RF: MAE = 1308, CV_STD_MAE = 9.40
model = min(mean_mae, key=mean_mae.get)
print('\nBest model with the lowest MAE:')
print(model)

###########################
# Part 4 - DEPLOY PHASE ###
###########################
# Model Validation with Regression Plots:
# Fit the best model:
model.fit(X_train, y_train.ravel())

# Make predictions:
y_pred_var_med_weekly = model.predict(X_val)
y_val = y_val.to_frame() #Sales Variance = [Median - Weekly Sales]
y_val['Sales_Var_Med_Weekly_Pred'] = y_pred_var_med_weekly #Sales Variance Predicted

df_best = pd.merge(df_final_train, y_val[['Sales_Var_Med_Weekly_Pred']], how='left', 
                  left_index=True, right_index=True, suffixes=['_True','_Pred'])

# Remove rows where Sales_Var_Med_Weekly_Pred is Null:
df_best = df_best[~pd.isnull(df_best['Sales_Var_Med_Weekly_Pred'])]
df_best.head()

# Derive the Weekly Sales Prediction: Median Sales - (Median Sales - Weekly Sales)_Pred
df_best['Weekly_Sales_Pred'] = df_best['Median_Sales']-df_best['Sales_Var_Med_Weekly_Pred']

# Plot a comparison scatter plot: predicted vs. actual
plt.figure(figsize=(14,7))
plt.subplots_adjust(wspace=0.2)
plt.subplot(1,2,1)
plot_reg_compare(df_best['Weekly_Sales_Pred'], df_best['Weekly_Sales'])
plt.title('Weekly Sales of Predicted vs. Actual: RFRegressor')

# Plot a residual scatter plot: predicted vs. residual
plt.subplot(1,2,2)
plot_reg_residual(df_best['Weekly_Sales_Pred'], df_best['Weekly_Sales'])
plt.title('Weekly Sales of Predicted vs. Residual: RFRegressor')
plt.show()

# Compute the Mean Absolute Error (MAE):
print("Medians: "+str(sum(abs(df_best['Sales_Var_Median_Weekly']))/df_best.shape[0]))
print("Best Model: "+str(sum(abs(df_best['Weekly_Sales']-df_best['Weekly_Sales_Pred']))/df_best.shape[0]))

del(df_best)

# --- 17. Automate the model pipeline --- #
# make predictions based on a test set
df_pred = model.predict(df_final_test[selection])
df_pred = pd.DataFrame(df_pred)
df_pred.columns = ['Sales_Var_Med_Weekly_Pred']

# make predictions dataframe:
df_predictions = pd.concat([df_final_test.reset_index(drop=True), df_pred], axis=1)
df_predictions['Weekly_Sales_Pred'] = df_predictions['Median_Sales']-df_predictions['Sales_Var_Med_Weekly_Pred']
df_predictions = df_predictions[['Store', 'Dept', 'Date_String', 'Weekly_Sales_Pred']]

del(df_pred)

df_predictions = gen_predictions_file(df_predictions, 'Store', 'Dept', 'Date_String')
df_predictions.head()


# --- 18. Deploy the solution --- #
#store feature importances
if hasattr(model, 'feature_importances_'):
    importances = model.feature_importances_
else:
# linear models don't have one:
    importances = [0]*len(df_final_train.columns)

feature_importances = pd.DataFrame({'feature':df_final_train[selection].columns,
                                    'importance':importances})
feature_importances.sort_values(by='importance', ascending=False, inplace=True)
    
#set index to 'feature'
feature_importances.set_index('feature', inplace=True, drop=True)

#create a plot
feature_importances[0:15].plot.bar(align='center')
plt.xticks(rotation=45, fontsize=9)
plt.title('Feature Importance Plot: Best Model')
plt.show()
    
#Save model results as .csv file:
model_results(model, mean_mae[model], df_predictions, feature_importances)