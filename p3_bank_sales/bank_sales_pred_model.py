# Import and set directory:
import os
os.getcwd()

# Set path:
path = 'D:/TY Workspace/3. Data Science Portfolio/1. Completed Portfolios/P3_Bank_Sales_Prediction/2_Development'
os.chdir(path)

# Import all libraries #
import itertools
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

        
def drop_column(df, var_name):
    ''' drop a column on dataframe '''
    df = df.drop(var_name, axis=1)
    return df


def clean_data(raw_df):
    '''remove rows that contain outliers'''
    clean_df = raw_df.drop_duplicates(subset='ID')
    return clean_df
    

def join_data(df1, df2, join_type, key=None,
              left_index=None, right_index=None):
    '''merge the dataframes by a key'''
    df_join = pd.merge(df1, df2, how=join_type, on=key,
                       left_index=False, right_index=False)
    return df_join


def convert_DOB_to_age(df, var_name, date_format):
    '''set DOB as datetime and calculate age'''
    df[var_name] = pd.to_datetime(df[var_name], format=date_format)
    df[var_name] = np.where(df[var_name] > datetime.now(),
                            df[var_name] - timedelta(365*100),
                            df[var_name])
    df['age'] = datetime.now().year - df[var_name].apply(lambda x:x.year)


def convert_date_to_columns(df, var_name, date_format):
    '''create day, weekday, month, year from Lead_Creation_Date'''
    df[var_name] = pd.to_datetime(df[var_name], format=date_format)
    df['LCD_day'] = df[var_name].apply(lambda x:x.day)
    df['LCD_weekday'] = df[var_name].apply(lambda x:x.weekday())
    df['LCD_month'] = df[var_name].apply(lambda x:x.month)
    df['LCD_year'] = df[var_name].apply(lambda x:x.year)


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


def apply_binning(df, new_var, old_var, bins, labels):
    '''apply binning on a selected variable'''
    df[new_var] = pd.cut(df[old_var], bins=bins,
                          labels=labels, include_lowest=True)
    return df[new_var]


def one_hot_encode_feature(df, cat_vars=None, num_vars=None):
    '''performs one-hot encoding on all categorical variables and
       combine results with numerical variables '''
    cat_df = pd.get_dummies(df[cat_vars], drop_first=True)
    num_df = df[num_vars].apply(pd.to_numeric)
    return pd.concat([cat_df, num_df], axis=1)


def get_label_data(df, label_var):
    '''separate label from a dataframe'''
    df_label = df[label_var]
    return df_label


def feature_scale_data(X):
    '''Feature scaled data based on standardization'''
    sc_X = StandardScaler()
    X_std = sc_X.fit_transform(X)
    return X_std
    

def score_model_roc_auc(model, X_train, y_train, X_val, y_val):
    '''computes the roc_auc score for probability of being a stroke case'''
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_val)
    return skm.roc_auc_score(y_val, probs[:,1])


def model_tuning_param(model, feature_df, label_df, param_dist, n_iter):
    '''performs RandomizedSearchCV to tune model hyper-parameters'''
    random_search = RandomizedSearchCV(model, param_dist, n_iter, cv=5)
    random_search.fit(feature_df, label_df)
    return random_search


def print_best_param(random_search, param_1=None, param_2=None, param_3=None, param_4=None):
    '''print the best model parameter(s)'''
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
    '''print out the model performances'''
    print('\nModel:\n', model)
    print('Average roc_auc:\n', mean_roc_auc[model])
    print('Std. Dev during CV:\n', cv_std[model])    


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
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i,j], fmt), 
                 horizontalalignment='center', 
                color='white' if cm[i,j] > thresh else 'black')
    plt.tight_layout()
    plt.xlabel('predicted label')
    plt.ylabel('true label')


def report_class_summary(model_name, y_act, y_pred):
    '''Write a classification summary report''' 
    print ('Accuracy of ' + model_name + ' is %0.2f'% skm.accuracy_score(y_act, y_pred))
    print ('Precision of ' + model_name + ' is %0.2f'% skm.precision_score(y_act, y_pred))
    print ('Recall of ' + model_name + ' is %0.2f'% skm.recall_score(y_act, y_pred))
    print ('ROC score of ' + model_name + ' is %0.2f'% skm.roc_auc_score(y_act, y_pred))
    
    
def model_results(model, mean_roc_auc, predictions, feature_importances):
    '''saves the model name, mean_roc_auc, predicted rate, and feature importances'''
    with open('model.txt', 'w') as file:
        file.write(str(model))
        feature_importances.to_csv('feat_importances.csv')
        predictions.to_csv('pred_results_best.csv', index=False)

# --- 2. Load the data --- #
# define input CSVs:
if __name__ == '__main__':
    train_file = 'bank_sales_train.csv'
    test_file = 'bank_sales_test.csv'

# Load data
df_train = load_file(train_file)
df_test = load_file(test_file)

# define variables:
# -- Note: Var1 and Employer_Category2 needs to be re-encoded --
label_var = 'Approved'
id_var = 'ID'
list_vars = ['DOB', 'Lead_Creation_Date']

# check data types on dataframes:
df_train.info()

# check top 10 rows:
df_train.head()

# create a dataframe with label:
df_label = df_train[['ID', 'Approved']]

# drop a specified column:
df_train = drop_column(df_train, label_var)

# join a train set and label:
train_raw_df = join_data(df_train, df_label, 'inner', key='ID')

# --- 5. ETL - cleaning data --- #
# clean data:
clean_train_df = shuffle(clean_data(train_raw_df)).reset_index(drop=True)
clean_test_df = shuffle(clean_data(df_test)).reset_index(drop=True)

del(train_raw_df, df_train, df_test)

# create a dataframe: test set 'ID'
df_test_id = clean_test_df['ID']

# --- 6. Feature Encoding --- #
#list(clean_train_df['Var1'].unique())
#list(clean_train_df['Employer_Category2'].unique())

# define mapping for ordinal and nominal features:
map_var1 = {0:'Level 1', 2:'Level 2', 4:'Level 3', 7:'Level 4', 10:'Level 5'} # Ordinal
map_Emp_Category2 = {1:'A', 2:'B', 3:'C', 4:'D'} # Nominal

# encode features into to mapping values: train set
clean_train_df['Var1'] = encode_categorical_feature(clean_train_df,'Var1',map_var1)
clean_train_df['Employer_Category2'] = encode_categorical_feature(clean_train_df,'Employer_Category2',map_Emp_Category2)

# encode features into to mapping values: test set
clean_test_df['Var1'] = encode_categorical_feature(clean_test_df,'Var1',map_var1)
clean_test_df['Employer_Category2'] = encode_categorical_feature(clean_test_df,'Employer_Category2',map_Emp_Category2)
del(map_var1, map_Emp_Category2)

# binary encoding: missing features
#missing_vars = ['Loan_Amount', 'EMI', 'Interest_Rate']

# compute binary flag: 0 or 1 on train and test sets


# --- 7. Feature Imputation --- #
# check missing: before imputation
df_missing_pre = EDA_missing_data(clean_train_df)
df_missing_pre

# feature imputation:
feature_replacement(clean_train_df)
feature_replacement(clean_test_df)

# check missing: after imputation
df_missing_post = EDA_missing_data(clean_train_df)
df_missing_post

# --- 8. Feature Engineering --- #
# create age from DOB: train and test sets
date_format = '%d/%m/%y'
convert_DOB_to_age(clean_train_df, 'DOB', date_format)
convert_DOB_to_age(clean_test_df, 'DOB', date_format)

# create from Lead_Creation_Date: day, weekday, month, year
convert_date_to_columns(clean_train_df, 'Lead_Creation_Date', date_format)    
convert_date_to_columns(clean_test_df, 'Lead_Creation_Date', date_format)    

# --- average Loan_Amount by categorical variables --- #
df_avg_loan_amt_by_primary_bank_type = avg_groupby_data(clean_train_df, 'Loan_Amount', 'Primary_Bank_Type', 'avg_loan_amt_primary_bank_type')
df_avg_loan_amt_by_employer_cat = avg_groupby_data(clean_train_df, 'Loan_Amount', 'Employer_Category1', 'avg_loan_amt_employer_cat')
#df_avg_loan_amt_by_source_cat = avg_groupby_data(clean_train_df, 'Loan_Amount', 'Source_Category', 'avg_loan_amt_source_cat')

# perform left joins on avg grouped dataframes:
clean_train_df = join_data(clean_train_df, df_avg_loan_amt_by_primary_bank_type, 'left', key='Primary_Bank_Type')
clean_train_df = join_data(clean_train_df, df_avg_loan_amt_by_employer_cat, 'left', key='Employer_Category1')
#clean_train_df = join_data(clean_train_df, df_avg_loan_amt_by_source_cat, 'left', key='Source_Category')

clean_test_df = join_data(clean_test_df, df_avg_loan_amt_by_primary_bank_type, 'left', key='Primary_Bank_Type')
clean_test_df = join_data(clean_test_df, df_avg_loan_amt_by_employer_cat, 'left', key='Employer_Category1')
#clean_test_df = join_data(clean_test_df, df_avg_loan_amt_by_source_cat, 'left', key='Source_Category')

del(df_avg_loan_amt_by_primary_bank_type, df_avg_loan_amt_by_employer_cat)

# --- 9. Exploratory Data Analysis --- #
# convert data type as category:
convert_dt_as_category(clean_train_df)
convert_dt_as_category(clean_test_df)

# save the cleaned train dataframe for EDA:
df_eda = clean_train_df
df_eda.to_csv('bank_sales_eda.csv', index=False, index_label=None)
del(df_eda)

# list of categorical and numerical features:
unwanted_list = {'ID', 'Employer_Code', 'City_Code', 'Source', 'Customer_Existing_Primary_Bank_Code'}
cat_feat = list(clean_train_df.select_dtypes(include='category').columns)
cat_feat = [item for item in cat_feat if item not in unwanted_list]

num_var_disc = list(clean_train_df.select_dtypes(include='int64').columns)
num_var_cont = list(clean_train_df.select_dtypes(include='float64').columns)

# concatenate numerical feature list:
num_feat = num_var_disc + num_var_cont
num_feat.remove('Approved')

del(num_var_disc, num_var_cont)

# perform summary statistics: numerical
df_stat_num = EDA_summary_stat_num(clean_train_df[num_feat])
df_stat_num

# perform summary statistics: categorical
df_stat_cat = EDA_summary_stat_cat(clean_train_df[cat_feat])
df_stat_cat

# --- 10. Prepare Training Data --- #
# one-hot encoding and concatenate numerical & categorical:
feature_df = one_hot_encode_feature(clean_train_df, cat_vars=cat_feat, num_vars=num_feat)
test_df = one_hot_encode_feature(clean_test_df, cat_vars=cat_feat, num_vars=num_feat)

# retrieve a label: 
label_df = clean_train_df['Approved']

# delete dataframes:
del(clean_train_df, clean_test_df)

# compute % loan approval:
label_pct = pd.DataFrame(df_label.groupby('Approved')['Approved'].count())
label_pct['pct_Approved'] = (label_pct['Approved']/len(df_label)) * 100
label_pct['pct_Approved']
del(df_label)

# --- 11. split data into train and test set --- # 
X_train, X_val, y_train, y_val = train_test_split(feature_df, label_df, test_size=1/4,
                                                    random_state=0, stratify=label_df)

# --- SMOTE: oversample on minority class label --- #
sm = SMOTE(random_state=0, ratio='minority', n_jobs=-1)
X_train_resampled, y_train_resampled = sm.fit_sample(X_train, y_train)

# count rows on resampled label:
pd.value_counts(y_train_resampled)

# --- 12. Feature seleciton: Feature Importance --- # 
baseline = RandomForestClassifier(n_estimators=100, n_jobs=-1)
baseline.fit(X_train_resampled, y_train_resampled)

# predict probablities and compute roc_auc score on postivie cases:
probs = baseline.predict_proba(X_val)
skm.roc_auc_score(y_val, probs[:,1]) 

# baseline: test predictions
test_probs = baseline.predict_proba(test_df)

# save a dataframe: test results
results = pd.DataFrame({'ID':df_test_id,
                        'Approved':test_probs[:,1]})
    
results.to_csv('prob_results_baseline.csv', index=False, index_label=None)

del(test_probs, results)

############################
# Part 3 - DEVELOP PHASE ###
############################
# --- 13. create models --- # 
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

# print the best model parameters: LogisticRegression    
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

# print the best model parameters: LogisticRegression    
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

# print the best model parameters: LogisticRegression    
param_1 = 'n_estimators'
param_2 = 'max_depth'
param_3 = 'colsample_bytree'
param_4 = 'learning_rate'

print_best_param(random_search_xgb, param_1, param_2, param_3, param_4)
###############################################################################    

# --- 14. cross-validate models --- # 
# a list of models to train: 5 fold cross-validation
lr = LogisticRegression(tol=4.308e-05, C=7, penalty='l1', solver='liblinear')
tree = DecisionTreeClassifier(max_depth=95, min_samples_split=7, min_samples_leaf=2,
                              max_features=6, random_state=0)
forest = RandomForestClassifier(n_estimators=39, max_depth=77, min_samples_split=8,
                                min_samples_leaf=2, random_state=0) 
xgb = XGBClassifier(n_estimators=49, max_depth=99, colsample_bytree=0.100846,
                    learning_rate=0.20463, random_state=0)

# a list of classifiers:
models.extend([lr, tree, forest, xgb])

# cross-validate models, using roc_auc to evaluate and print the summaries
print('begin cross-validation')
for model in models:
    model_train(model, X_train_resampled, y_train_resampled, n_proc, mean_roc_auc, cv_std)
    model_summary(model, mean_roc_auc, cv_std)
    
# --- 15. select the best model --- #
model = max(mean_roc_auc, key=mean_roc_auc.get)
print('\nBest model with the highest mean roc_auc:')
print(model)

# --- compute a roc_auc score on "positive cases" --- #
#score_model_roc_auc(lr, X_train_resampled,
#                    y_train_resampled, X_val, y_val)

# --- 16. Model Evaluation ---    
# compute predictions by a model:
lr.fit(X_train_resampled, y_train_resampled)
y_pred_lr = lr.predict(X_val)

tree.fit(X_train_resampled, y_train_resampled)
y_pred_tree = tree.predict(X_val)

forest.fit(X_train_resampled, y_train_resampled)
y_pred_forest = forest.predict(X_val)

xgb.fit(X_train_resampled, y_train_resampled)
y_pred_xgb = xgb.predict(X_val.values)

# compute confusion matrices by a model:
cm_lr = compute_confusion_matrix(y_val, y_pred_lr)

cm_tree = compute_confusion_matrix(y_val, y_pred_tree)

cm_forest = compute_confusion_matrix(y_val, y_pred_forest)

cm_xgb = compute_confusion_matrix(y_val, y_pred_xgb)

# define class labels for Approved:
class_labels = np.array(['Non-Approved', 'Approved'], dtype=str)

#####################################################
# Confusion Matrix & Classification Metrics Summary #
#####################################################
# --- Model 1 --- #
# Plot a confusion matrix: 
plot_confusion_matrix(cm_lr, class_labels)
plt.title('Normalized Confusion Matrix: Logistic Regression')
plt.show()

# Report classification metrics summary:
report_class_summary('Logistic Regression', y_val, y_pred_lr)

# --- Model 2 --- #
# Plot a confusion matrix: 
plot_confusion_matrix(cm_tree, class_labels)
plt.title('Normalized Confusion Matrix: Decision Tree')
plt.show()

# Report classification metrics summary:
report_class_summary('Decision Tree', y_val, y_pred_tree)

# --- Model 3 --- #
# Plot a confusion matrix: 
plot_confusion_matrix(cm_forest, class_labels)
plt.title('Normalized Confusion Matrix: Random Forest')
plt.show()

# Report classification metrics summary:
report_class_summary('Random Forest', y_val, y_pred_forest)

# --- Model 4 --- #
# Plot a confusion matrix: 
plot_confusion_matrix(cm_xgb, class_labels)
plt.title('Normalized Confusion Matrix: XGBoost')
plt.show()

# Report classification metrics summary:
report_class_summary('XGBoost', y_val, y_pred_xgb)

# --- 17. Model Plotting --- 
# ROC for each classifiers
clf_labels = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'XGBoost']
all_clf = [lr, tree, forest, xgb]

# plot a ROC_AUC curve:
plt.figure(figsize=(8,8))
colors = ['green', 'blue', 'orange', 'red']
linestyles = ['-', ':', ':', ':'] 
for clf, label, clr, ls \
    in zip(all_clf, clf_labels, colors, linestyles):  
        # Assume the label of the positive class is 1
        y_pred = clf.fit(X_train_resampled,
                         y_train_resampled).predict_proba(X_val.values)[:,1]
        fpr, tpr, thresholds = skm.roc_curve(y_true=y_val, y_score=y_pred)
        roc_auc = skm.auc(x=fpr, y=tpr)
        plt.plot(fpr, tpr, color=clr, linestyle=ls,
                 label='%s (AUC = %0.2f)' % (label, roc_auc))
plt.plot([0,1], [0,1], linestyle='--', color='gray', linewidth=2)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.grid(alpha=0.5)
plt.title('ROC Curve: Evaluation of Classifiers [test set]')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right', prop={'size': 15})
plt.show()
 
## compute avg. precision score:
#y_score_lr = lr.fit(X_train_resampled, y_train_resampled).predict_proba(X_val.values)[:,1]
#avg_precision = skm.average_precision_score(y_val, y_score_lr)
#print('Average P-R score of Logistic Regression: {0:0.2f}'.format(avg_precision))
#
## plot a P-R curve:
#precision, recall, _ =skm.precision_recall_curve(y_val, y_score_lr)
#plt.step(recall, precision, color='navy', where='post', label='Precision-Recall Curve')
#plt.title('P-R Curve: LogisiticRegression [test set] AP={0:0.2f}'.format(avg_precision))
#plt.xlabel('Recall (Sensitivity)')
#plt.ylabel('Precision (PPV)')
#plt.ylim([0, 1.05])
#plt.xlim([0, 1.0])
#plt.legend(loc='lower right')
#plt.show()

###########################
# Part 4 - DEPLOY PHASE ###
###########################
# --- 18. automate the model pipeline --- #
# make predictions on a test set:
df_pred_probs = lr.predict_proba(test_df)

# make predictions dataframe:
results = pd.DataFrame({'ID':df_test_id,
                        'Approved':df_pred_probs[:,1]})
results.to_csv('prob_results_best.csv', index=False, index_label=None)

# --- 19. deploy the solution --- #
#store feature importances
if hasattr(model, 'feature_importances_'):
    importances = model.feature_importances_
else:
# linear models don't have feature_importances_
    importances = [0]*len(X_train.columns)

# sort a dataframe by feature importance score: 
feature_importances = pd.DataFrame({'feature':X_train.columns,
                                        'importance':importances})
feature_importances.sort_values(by='importance', inplace=True, ascending=False)

# set index to 'feature'
feature_importances.set_index('feature', inplace=True, drop=True)

# create a bar plot:    
feature_importances[0:15].plot.bar(align='center')
plt.title('feature importance plot: best trained model')
plt.xticks(rotation=270, fontsize=6)
plt.show()
    
## save model results as .csv file:
#model_results(model, mean_roc_auc[model], results, feature_importances)