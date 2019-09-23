# Import and set directory:
import os
os.getcwd()

# Set path:
#path = 'D:/P_Poverty_Prediction/A. Coding'
path = 'D:/TY Workspace/3. Data Science Portfolio/p6_bank_sales_prediction/A. Coding'
os.chdir(path)

# Import all libraries #
import sys
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
plt.style.use('ggplot')
import operator
from itertools import cycle
import scipy.stats as sp
from scipy import interp
from sklearn.externals import six
from sklearn.pipeline import _name_estimators
import sklearn.metrics as skm
import sklearn.base as skb
from sklearn.utils import shuffle, resample
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from sklearn.preprocessing import Imputer, StandardScaler, LabelEncoder
from sklearn.model_selection import KFold, train_test_split
from sklearn.model_selection import cross_val_predict, cross_val_score, RandomizedSearchCV
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# Authorship:
__author__ = 'Taesun Yoo'
__email__ = 'yoots1988@gmail.com'

# Check out Python version:
print ("Python version: {}".format(sys.version))

#############################
# Part 2 - DISCOVER PHASE ###
#############################
# --- 2. Write Out List of Functions --- #
def load_file(file):
    '''load CSV file as a dataframe'''
    df = pd.read_csv(file)
    return df

def drop_column_by_index(df, var_name):
    '''drop a column by index'''
    df = df.drop(var_name, axis=1)
    return df

def join_data(df_train, df_label, key,
              left_index=None, right_index=None):
    '''merge the feature(s) and label dataframes'''
    df_join = pd.merge(df_train, df_label, how='inner', on=key,
             left_index=False, right_index=False)
    return df_join    

def clean_data(df):
    '''drop any duplicates based on a specific column'''
    clean_df = df.drop_duplicates(subset='ID')
    return clean_df

def convert_DOB_to_age(df, var_name, date_format):
    '''set DOB as datetime and calculate age'''
    df[var_name] = pd.to_datetime(df[var_name], format=date_format)
    df[var_name] = np.where(df[var_name] > datetime.now(),
                           df[var_name] - timedelta(365*100),
                           df[var_name])
    df['Age'] = datetime.now().year - df[var_name].apply(lambda x:x.year)

def convert_date_to_columns(df, var_name, date_format):
    '''use lead creation date to create year, month, day and weekday'''
    df[var_name] = pd.to_datetime(df[var_name], format=date_format)
    df['LCD_year'] = df[var_name].apply(lambda x:x.year)
    df['LCD_month'] = df[var_name].apply(lambda x:x.month)
    df['LCD_day'] = df[var_name].apply(lambda x:x.day)
    df['LCD_weekday'] = df[var_name].apply(lambda x:x.weekday())    

def convert_dt_as_category(df):
    '''convert metadata type from object to category'''
    for var_name in df.columns:
        if df[var_name].dtype.name == 'object':
            df[var_name] = df[var_name].astype('category')

def convert_dt_as_object(df):
    '''convert metadata type from category to object'''
    for column in df.columns:
        if df[column].dtype.name == 'category':
            df[column] = df[column].astype('object')

def EDA_missing_data(df):
    '''compute missing % on feature column(s)'''
    missing_df = pd.DataFrame(df.isnull().sum())
    missing_df.columns = ['count']
    missing_df = missing_df.sort_values(by='count', ascending=False)
    missing_df['pct'] = missing_df['count']/len(df)
    return missing_df

def EDA_summary_stat_num(df):
    '''compute summary statistics on numerical variables'''
    df_stat_num = df.describe().T
    df_stat_num = df_stat_num[['count', 'min', 'mean', 'max', '25%', '50%', '75%', 'std']]
    df_stat_num = df_stat_num.sort_values(by='count', ascending=True)
    df_stat_num = pd.DataFrame(df_stat_num)
    return df_stat_num

def EDA_summary_stat_cat(df):
    '''compute summary statistics on categorical variables'''    
    df_stat_cat = pd.DataFrame(df.describe(include='O').T)
    return df_stat_cat

def EDA_plot_correlation(df_EDA):
    '''compute and plot correlation matrix'''
    corr = df_EDA.corr()
    # Create a mask to filter matrix: diagonally
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    # Matrix Plot:
    fig, ax = plt.subplots(figsize=(7,7))
    cmap = sns.diverging_palette(220,10,as_cmap=True)
    sns.set(font_scale=1.1)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                annot=True, square=True, linewidths=.5, fmt=".2f",
                annot_kws={'size':10}, cbar_kws={'shrink':.6})
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

def EDA_outliers(df_stat_num):
    '''Compute outliers based on summary stat of numerical vars'''
    df_stat_num['IQR'] = df_stat_num['75%'] - df_stat_num['25%']
    df_stat_num['UB'] = df_stat_num['75%'] + 1.5*df_stat_num['IQR']
    df_stat_num['LB'] = df_stat_num['25%'] - 1.5*df_stat_num['IQR']
    df_outliers = df_stat_num[['LB', 'min', 'UB', 'max']]
    return df_outliers

def encode_categorical_feature(df, var_name, map_name):
    '''encode categorical features into mapped values'''
    df[var_name] = df[var_name].map(map_name)
    return df[var_name]

def Binary_Encode_Missing_Feature(df, missing_var_list):
    '''Binary encode missing feature(s) as a flag 0:no or 1:yes'''
    for var in missing_var_list:
        df['Is_Missing_' + var] = np.where(df[var].isnull(), 1, 0)

def split_dataframe(df):
    '''Split dataframe by features and a label'''
    X, y = df.drop(['Approved'], axis=1), df['Approved']
#    X, y = df.iloc[:, :-1], df.iloc[:,-1]
    return X, y

def feature_imputer(X, missing_val_format, method, indices):
    '''impute missing values based on uni-variate methods for int/float'''
    imputer = Imputer(missing_values=missing_val_format, strategy=method, axis=0)
    imputer = imputer.fit(X.iloc[:, indices])
    X.iloc[:, indices] = imputer.transform(X.iloc[:, indices])
    return X.iloc[:, indices]
    
def missing_val_replacement(X):
    '''replace missing values based on specific data type and a column'''
    for column in X.columns:
        if X[column].dtype.name == 'object':
            mode = X[column].mode()
            X[column] = X[column].fillna(str(mode))
        elif X[column].dtype.name == 'float64':
            mean = X[column].mean()
            X[column] = X[column].fillna(mean)
        elif X[column].dtype.name == 'datetime64[ns]':
            pseudo_date = pd.Timestamp.max
            X[column] = X[column].fillna(pseudo_date)
        else: 
            X['Age'] = X['Age']
            median = X['Age'].median()
            X['Age'] = X['Age'].fillna(median)

def one_hot_encode_feature(df, cat_vars=None, num_vars_disc=None, num_vars_con=None):
    '''perform one-hot encoding on all features and concatenate them'''
    cat_df = pd.get_dummies(df[cat_vars], drop_first=True)
    num_disc_df = pd.get_dummies(df[num_vars_disc], drop_first=True)
    num_con_df = pd.get_dummies(df[num_vars_con], drop_first=True)
    return pd.concat([cat_df, num_disc_df, num_con_df], axis=1)

def label_encode_feature(df, var_name):
    '''perform label encoding on categorical features'''
    label_encoder = LabelEncoder()
    label_encoder.fit(df[var_name])
    df[var_name+'_enc'] = label_encoder.transform(df[var_name])
    return label_encoder

def model_tuning_param(model, feature_df, label_df, param_dist, n_iter):
    '''performs RandomizedSearchCV to tune hyper-parameters'''
    random_search = RandomizedSearchCV(model, param_dist, n_iter, cv=5)
    random_search.fit(feature_df, label_df)
    return random_search

def print_best_param(random_search, param_1=None, param_2=None, 
                     param_3=None, param_4=None):
    '''print the best model parameters'''
    print("Best " + param_1 + ":", random_search.best_estimator_.get_params()[param_1])
    print("Best " + param_2 + ":", random_search.best_estimator_.get_params()[param_2])
    print("Best " + param_3 + ":", random_search.best_estimator_.get_params()[param_3])
    print("Best " + param_4 + ":", random_search.best_estimator_.get_params()[param_4])

def model_train(model, feature_df, label_df, n_proc, mean_roc_auc, cv_std):
    '''train a model and output mean roc_auc and CV std.dev roc_auc'''
    roc_auc = cross_val_score(model, feature_df, label_df, n_jobs=n_proc,
                              cv=5, scoring='roc_auc')
    mean_roc_auc[model] = np.mean(roc_auc)
    cv_std[model] = np.std(roc_auc)

def model_summary(model, mean_roc_auc, cv_std):
    print('\nModel:\n', model)
    print('Average roc_auc:\n', mean_roc_auc[model])
    print('Std. Dev during CV:\n', cv_std[model])

def model_results(model, mean_roc_auc, predictions, feature_importances):
    '''save the model name, mean_roc_auc, predictions, and feature importances'''
    with open('model.txt', 'w') as file:
        file.write(str(model))
        feature_importances.to_csv('feat_importances.csv')
        predictions.to_csv('pred_results_best.csv', index=False)

def score_model_roc_auc(model, X_train, y_train, X_val, y_val):
    '''compute the roc_auc score for Approved cases '''
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_val)
    return skm.roc_auc_score(y_val, probs[:, 1])
        
# --- 3. Load the data --- #
# Define input CSVs:
if __name__ == '__main__':
    train_file = 'bank_sales_train.csv'
    test_file = 'bank_sales_test.csv'

# Load data
df_train = load_file(train_file)
df_test = load_file(test_file)

# Define type of variables list:
#df_train.select_dtypes(include='object').columns #df_train.select_dtypes(include='int64').columns 
#df_train.select_dtypes(include='float64').columns
# Note: 'Var1', 'Employer_Category2' are categorical variables need to be re-encoded

label_var = 'Approved'

id_var = 'ID'

drop_vars = ['DOB', 'Lead_Creation_Date']

# Create a label dataframe:
df_label = df_train[['ID', 'Approved']]

# Drop a column by index: poverty_rate
df_train = drop_column_by_index(df_train, label_var)

# join train set and label:
train_raw_df = join_data(df_train, df_label, 'ID')

# --- 4. Perform data cleaning and quality check --- #
# Clean invalid data and duplicates: train and test set
clean_train_df = shuffle(clean_data(train_raw_df)).reset_index(drop=True)
clean_test_df = shuffle(clean_data(df_test)).reset_index(drop=True)
del(train_raw_df, df_train, df_test)

# --- 5. Feature Engineering --- #
# Create 'Age' from DOB on train and test set:
date_format = "%d/%m/%y"
convert_DOB_to_age(clean_train_df, 'DOB', date_format)    
convert_DOB_to_age(clean_test_df, 'DOB', date_format)    

# Create Lead_Creation_Date into year, month, day, weekday
convert_date_to_columns(clean_train_df, 'Lead_Creation_Date', date_format)    
convert_date_to_columns(clean_test_df, 'Lead_Creation_Date', date_format)    

clean_train_df.columns
clean_train_df.dtypes

# Define list of variables: categorical and numerical
cat_vars = list(clean_train_df.select_dtypes(include='object').columns)
cat_vars.remove('ID')

num_vars_con = list(clean_train_df.select_dtypes(include='float64').columns)
num_vars_disc = list(clean_train_df.select_dtypes(include='int64').columns)
num_vars_disc.remove('Approved')

# Compute missing value % on a dataframe:
missing_df_train = EDA_missing_data(clean_train_df)
missing_df_train

missing_df_test = EDA_missing_data(clean_test_df)
missing_df_test

# --- 6. Explore the data (EDA) --- # 
# Compute summary statistic
df_stat_num = EDA_summary_stat_num(clean_train_df[num_vars_con])
df_stat_num

df_stat_cat = EDA_summary_stat_cat(clean_train_df[cat_vars])
df_stat_cat

# Check 1: Drop rows with above index list
###############################################################################
# Compute indexes where invalid values occurred on selected column:
# Drop rows with above index list:
###############################################################################
# save 'ID' from test set:
df_id_test = clean_test_df[id_var]

## Correlation matrix plot # 
############################
#df_EDA = clean_train_df.copy()
#df_EDA = drop_column_by_index(df_EDA, cat_vars)
#df_EDA = drop_column_by_index(df_EDA, id_var)
#
## CM Plot:
#EDA_plot_correlation(df_EDA)
#plt.title('Correlation Matrix on Training Set')
#plt.show()
#
## Compute the correlation of each feature against poverty rate: order of magnitude
#CM_summary = df_EDA.corr().sort_values(by=['Approved'], ascending=False)['Approved']
#print(CM_summary)

# Check 2: Drop a column where missing data is greater or equal to 50%
###############################################################################
# Compute a column list with missing more than 50%:
# Remove 'EMI' & 'Interest_Rate':
missing_df_train[['pct']] > 0.5

# Drop column(s): missing value(s) > 50% and ID:
clean_train_df = drop_column_by_index(clean_train_df, 'ID')
clean_train_df = drop_column_by_index(clean_train_df, drop_vars)
#clean_train_df = drop_column_by_index(clean_train_df, 'EMI')
#clean_train_df = drop_column_by_index(clean_train_df, 'Interest_Rate')

clean_test_df = drop_column_by_index(clean_test_df, 'ID')
clean_test_df = drop_column_by_index(clean_test_df, drop_vars)
#clean_test_df = drop_column_by_index(clean_test_df, 'EMI')
#clean_test_df = drop_column_by_index(clean_test_df, 'Interest_Rate')

###############################################################################

# Check 3: handling outliers using IQR #
###############################################################################
# Compute IQR, LB, UB:
df_outliers = EDA_outliers(df_stat_num)
df_outliers

# Case 1: where feature's max is greater than UB value:
df_outliers[df_outliers['max'] > df_outliers['UB']]

# Case 2: where feature's min is lower than LB value:
df_outliers[df_outliers['min'] < df_outliers['LB']]

# --- 7. Feature encoding on categorical variables --- #
list(clean_train_df['Var1'].unique())
list(clean_train_df['Employer_Category2'].unique())

# Define manual mapping for Employer_Category2 & Var1:
var1_map = {0:'Level 1', 2:'Level 2', 4:'Level 3', 7:'Level 4', 10:'Level 5'} # Ordinal
Emp_Category2_map = {1:'A', 2:'B', 3:'C', 4:'D'} # Nominal

# Use mapping to transform the ordinal and nominal features:
clean_train_df['Var1'] = encode_categorical_feature(clean_train_df,'Var1',var1_map)
clean_train_df['Employer_Category2'] = encode_categorical_feature(clean_train_df,'Employer_Category2',Emp_Category2_map)

clean_test_df['Var1'] = encode_categorical_feature(clean_test_df,'Var1',var1_map)
clean_test_df['Employer_Category2'] = encode_categorical_feature(clean_test_df,'Employer_Category2',Emp_Category2_map)

# Encode feature(s) for: 'Loan Amount', 'EMI', 'Interest_Rate'
missing_var_list = ['Loan_Amount', 'EMI', 'Interest_Rate']

# Binary flag: 0 or 1 on training and test sets
Binary_Encode_Missing_Feature(clean_train_df, missing_var_list)
Binary_Encode_Missing_Feature(clean_test_df, missing_var_list)

# Inverse transform back to original categorical variables:
#inv_var1_map = {i:j for j, i in var1_map.items()}
#inv_Emp_Category2_map= {i:j for j, i in Emp_Category2_map.items()}

# --- 8. Feature imputation via univariate techniques --- #    
# Split data into input features and target variable #
X_train, y_train = split_dataframe(clean_train_df)
X_test = clean_test_df.copy()
del(clean_train_df, clean_test_df)

# check input features and target variable: train and test sets
print(X_train.head(), y_train.head())
X_train.isnull().sum()

# Imputation by mode, median, mean: train and test sets
# If categorical imputed by mode else imputed by mean;
missing_val_replacement(X_train)
missing_val_replacement(X_test)

# Check missing % on X_train:
X_train.isnull().sum()
X_test.isnull().sum()

# Save cleaned and imputed dataframe for EDA analysis:
EDA_df = pd.concat([X_train, y_train], axis=1)
EDA_df.to_csv('df_EDA.csv', index=False)

# concatenated imputed inputs and output label:
imputed_train_df = X_train.copy()
imputed_test_df = X_test.copy()
label_train_df = pd.DataFrame(y_train)
#del(X_train, X_test)

# check any missing values on imputed df:
imputed_train_df.isnull().sum()
imputed_test_df.isnull().sum()

# convert data types for correct metadata: train and test sets
# Fix DOB as datetime64[ns]
convert_dt_as_category(imputed_train_df)
convert_dt_as_category(imputed_test_df)

# check cleaned dataframe: data types
imputed_train_df.dtypes
imputed_test_df.dtypes

# --- 9. Feature engineering: groupby categorical var ---
# create groupby dataframes:

# perform left joins on avg_groupby_df:

# Note: When train and test set have difference # of unique values
# on categorical features then consider using "Label Encoder".

# --- 10. Label-encode on features --- # 
# Drop first dummy variable to avoid dummy variable trap on each converted feature!
cat_vars2 = list(imputed_train_df.select_dtypes(include='category').columns)

num_vars_disc2 = list(imputed_train_df.select_dtypes(include='int64').columns) 
#num_vars_disc2.remove('Approved')

num_vars_con2 = list(imputed_train_df.select_dtypes(include='float64').columns)

# label encoding on train and test sets:
feat_train_df = imputed_train_df.copy()
test_df = imputed_test_df.copy()
del(imputed_train_df, imputed_test_df)

for column in cat_vars2:
    label_encode_feature(feat_train_df, column)

for column in cat_vars2:
    label_encode_feature(test_df, column)

# List total number of encoded inputs and output:
feat_train_df.isnull().sum()
test_df.isnull().sum()

cat_cols_enc = [col_name + '_enc' for col_name in cat_vars2]
model_cols = cat_cols_enc + num_vars_disc2 + num_vars_con2

# Reform dataframe with encoded features: train and test set
enc_train_df = feat_train_df[model_cols] 
enc_test_df = test_df[model_cols]
del(feat_train_df, test_df)

enc_train_df.columns
enc_test_df.columns

# Compute ratio on Approved label:
label_pct = pd.DataFrame(label_train_df.groupby('Approved')['Approved'].count())
label_pct['pct'] = (label_pct['Approved']/len(label_train_df)) * 100

# Train_Test_Split:
X_train, X_val, y_train, y_val = train_test_split(enc_train_df, label_train_df, test_size=1/4,
                                                  random_state=0, stratify=label_train_df)

# --- 11. Resampling on non-approved cases: Over Sampling --- #
sm = SMOTE(random_state=0, ratio='minority', n_jobs=-1)
X_train_resampled, y_train_resampled = sm.fit_sample(X_train, y_train)

pd.value_counts(y_train_resampled)

# --- 12. variable inflation factor (VIF) testing --- # 

# --- 13. Feature seleciton: L1 regularization --- # 
# Save L1 feature selection results:
#Create L1 feature selection index:

# --- 14. Establish a baseline model --- # 
# Establish a baseline model:
baseline = RandomForestClassifier(n_estimators=100, n_jobs=-1)
baseline.fit(X_train_resampled, y_train_resampled)

probs = baseline.predict_proba(X_val)
predictions = baseline.predict(X_val)

skm.roc_auc_score(y_val, probs[:,1])

test_probs = baseline.predict_proba(enc_test_df)
results = pd.DataFrame({'ID':df_id_test,
                        'Approved':test_probs[:,1]})
results.to_csv('prob_results_baseline.csv', index=False, index_label=None)

############################
# Part 3 - DEVELOP PHASE ###
############################
# --- 15. Create models --- # 
# initialize model list and dicts
models = []
mean_roc_auc = {}
cv_std = {}
res = {}

# define common model parameters: num processors and shared model parameters
n_proc = 1
verbose_lv = 5

# create and tune the models that you brainstormed during part 2
###############################################################################        
# Hyper-parameters tuning: LogisticRegression
lr = LogisticRegression()
n_iter = 10
param_dist_lr = dict(tol=np.random.uniform(0, 0.0001, 10), C=sp.randint(1,50),
                     penalty=['l1', 'l2'], solver=['liblinear','saga'])
random_search_lr = model_tuning_param(lr, X_train_resampled, y_train_resampled, param_dist_lr, n_iter)

# print the best model parameters: LogisticRegression    
param_1 = 'tol'
param_2 = 'C'
param_3 = 'penalty'
param_4 = 'solver'

print_best_param(random_search_lr, param_1, param_2, param_3, param_4)
###############################################################################        
# Hyper-parameters tuning: DecisionTree
tree = DecisionTreeClassifier(criterion='gini', random_state=0)

n_iter = 10
param_dist_tree = {'max_depth': sp.randint(10,100),
                 'min_samples_split': sp.randint(2,10),
                 'min_samples_leaf': sp.randint(2,10),
                 'max_features': sp.randint(2,10)}

random_search_tree = model_tuning_param(tree, X_train_resampled, y_train_resampled, param_dist_tree, n_iter)

# print the best model parameters: DecisionTree    
param_1 = 'max_depth'
param_2 = 'min_samples_split'
param_3 = 'min_samples_leaf'
param_4 = 'max_features'

print_best_param(random_search_tree, param_1, param_2, param_3, param_4)
###############################################################################        
# Hyper-parameters tuning: RandomForest
forest = RandomForestClassifier(criterion='gini', random_state=0)

n_iter = 10
param_dist_forest = {'n_estimators': sp.randint(10,100),
                     'max_depth': sp.randint(10,100),
                     'min_samples_split': sp.randint(2,10),
                     'min_samples_leaf': sp.randint(2,10)}

random_search_forest = model_tuning_param(forest, X_train_resampled, y_train_resampled, param_dist_forest, n_iter)

# print the best model parameters: RandomForest    
param_1 = 'n_estimators'
param_2 = 'max_depth'
param_3 = 'min_samples_split'
param_4 = 'min_samples_leaf'

print_best_param(random_search_forest, param_1, param_2, param_3, param_4)
###############################################################################    
# Hyper-parameters tuning: XGBoost
xgb = XGBClassifier(random_state=0)
n_iter = 10

param_dist_xgb = {'n_estimators': sp.randint(10,100),
                 'max_depth': sp.randint(10,100),
                 'colsample_bytree': np.random.uniform(0, 1, 10),
                 'learning_rate': np.random.uniform(0, 1, 10)}

random_search_xgb = model_tuning_param(xgb, X_train_resampled, y_train_resampled, param_dist_xgb, n_iter)

# print the best model parameters: XGBoost    
param_1 = 'n_estimators'
param_2 = 'max_depth'
param_3 = 'colsample_bytree'
param_4 = 'learning_rate'

print_best_param(random_search_xgb, param_1, param_2, param_3, param_4)
###############################################################################   

# --- 16. Cross-validate models --- # 
#do 5-fold cross validation on models and measure roc_auc
# Model List to train: Order of Model Complexity
lr = LogisticRegression(tol=8.039e-05, C=24, penalty='l1', solver='liblinear')
tree = DecisionTreeClassifier(max_depth=20, min_samples_split=7, min_samples_leaf=2,
                              max_features=9, random_state=0)
forest = RandomForestClassifier(n_estimators=30, max_depth=56, min_samples_split=4,
                                min_samples_leaf=2, random_state=0) 
xgb = XGBClassifier(n_estimators=70, max_depth=75, colsample_bytree=0.38655442468838486,
                    learning_rate=0.12528436483791283, random_state=0)
# List of classifiers:
models.extend([lr, tree, forest, xgb])

# cross-validate models, using roc_auc to evaluate and print the summaries
print('begin cross-validation')
for model in models:
    model_train(model, X_train_resampled, y_train_resampled, n_proc, mean_roc_auc, cv_std)
    model_summary(model, mean_roc_auc, cv_std)

# --- 17. Select the best model: --- #
model = max(mean_roc_auc, key=mean_roc_auc.get)
print('\nBest model with the highest mean roc_auc:')
print(model)

# --- 18. compute roc_auc score on "Approved cases only" --- #
score_model_roc_auc(xgb, X_train_resampled, y_train_resampled,
                    X_val.values, y_val)

score_model_roc_auc(tree, X_train_resampled, y_train_resampled,
                    X_val.values, y_val)

score_model_roc_auc(forest, X_train_resampled, y_train_resampled,
                    X_val.values, y_val)

score_model_roc_auc(lr, X_train_resampled, y_train_resampled,
                    X_val.values, y_val)

###########################
# Part 4 - DEPLOY PHASE ###
###########################
# --- 19. Automate the model pipeline --- #
# make predictions based on a test set
df_test_selected = enc_test_df.copy()
df_pred_probs = model.predict_proba(df_test_selected)

# make predictions dataframe:
results = pd.DataFrame({'ID':df_id_test,
                        'Approved':df_pred_probs[:,1]})
results.to_csv('prob_results_best_model.csv', index=False, index_label=None)

# --- 20. Deploy the solution --- #
#store feature importances
if hasattr(model, 'feature_importances_'):
    importances = model.feature_importances_
else:
# linear models don't have feature_importances_
    importances = [0]*len(X_train.columns)

# Create a feature importance df and sort by importances:
feature_importances = pd.DataFrame({'feature':X_train.columns,
                                    'importances':importances})
feature_importances.sort_values(by='importances', inplace=True, ascending=False)
    
#set index to 'feature'
feature_importances.set_index('feature', inplace=True, drop=True)

#create a plot
feature_importances[0:15].plot.bar(align='center')
plt.title('Feature Importance Plot: best trained model')
plt.show()
    
#Save model results as .csv file:
model_results(model, mean_roc_auc[model], results, feature_importances)