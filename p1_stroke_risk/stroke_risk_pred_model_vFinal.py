# Import and set directory:
import os
os.getcwd()

# Set path:
#path = 'D:/P1_Stroke_Risk/A. Coding'
path = 'D:/TY Workspace/3. Data Science Portfolio/1. Completed Portfolios/p1_stroke_risk/A. Coding'
os.chdir(path)

# Import all libraries #
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

#############################
# Part 2 - DISCOVER PHASE ###
#############################

# --- 2. Write Out List of Functions --- #
def load_file(file):
    '''load the CSV files as a dataframe'''
    df = pd.read_csv(file)
    return df

def drop_column_by_index(df, var):
    '''drop a column by specified variable'''
    df = df.drop(var, axis=1)
    return df

def join_data(df_train, df_label, key, 
              left_index=None, right_index=None):
    '''Merge the feature and label dataframe(s)'''
    df_join = pd.merge(df_train, df_label, how='inner', on=key,
                         left_index=False, right_index=False)
    return df_join

def clean_data(df):
    '''drop any duplicate based on specific column'''
    clean_df = df.drop_duplicates(subset='id')
    return clean_df

def eda_missing_data(df):
    missing_df = pd.DataFrame(df.isnull().sum())
    missing_df.columns = ['count']
    missing_df['pct'] = (missing_df['count']/len(df))*100
    return missing_df

def eda_summary_stat_num(df):
    '''compute summary statistics for numerical variables'''
    df_stat_num = df.describe().T
    df_stat_num = df_stat_num[['count', 'min', 'mean', 'max', '25%', '50%', '75%', 'std']]
    df_stat_num = df_stat_num.sort_values(by='count', ascending=True)
    df_stat_num = pd.DataFrame(df_stat_num)
    return df_stat_num

def eda_summary_stat_cat(df):
    '''compute summary statistics for categorical variables'''
    df_stat_cat = pd.DataFrame(df.describe(include='O').T)
    return df_stat_cat

def compute_outliers(df_stat_num):
    df_stat_num['IQR'] = df_stat_num['75%'] - df_stat_num['25%']
    df_stat_num['UB'] = df_stat_num['75%'] + 1.5*df_stat_num['IQR']
    df_stat_num['LB'] = df_stat_num['25%'] - 1.5*df_stat_num['IQR']
    df_outliers = df_stat_num[['LB', 'min', 'UB', 'max']]
    return df_outliers

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
    plt.show()

def encode_categorical_feature(df, var_name, map_name):
    '''encode categorical features into mapping values'''
    df[var_name] = df[var_name].map(map_name)
    return df[var_name]

def feature_imputer(X, missing_val_format, method, indices):
    '''imputes missing values based on different uni-variate methods'''
    imputer = Imputer(missing_values=missing_val_format, strategy=method, axis=0)
    imputer = imputer.fit(X.iloc[:, indices])
    X.iloc[:, indices] = imputer.transform(X.iloc[:, indices])
    return X.iloc[:, indices]

def convert_data_type(df, var_name, dt_type):
    '''convert data type into specified metadata type'''
    df[var_name] = df[var_name].astype(dt_type)
    return df[var_name]

def split_dataframe(df):
    '''Split dataframe into features and label'''
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    return X, y

def avg_groupby_data(df, num_var, cat_var, avg_var_name):
    '''perform average group by categorical variable to compute a mean'''
    avg_groupby_val = df.groupby(cat_var)[num_var].mean().sort_values(ascending=False)
    avg_groupby_df = pd.DataFrame({cat_var:list(df[cat_var].unique()),
                                   avg_var_name:avg_groupby_val})
    avg_groupby_df.reset_index(drop=True, inplace=True)
    return avg_groupby_df

def left_join_data(train_df, avg_groupby_df, key=None, left_index=False, right_index=False):
    '''performs left join on train data to average groupby data'''
    joined_df = pd.merge(train_df, avg_groupby_df, how='left', on=key,
                         left_index=left_index, right_index=right_index)
    return joined_df

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

def split_data_by_age_group(df, var_name):
    '''split dataframe by age group'''
    df_age_group = pd.DataFrame(df.groupby(var_name)[var_name].count().sort_values(ascending=False))
    df_age_group.columns = ['count']
    df_age_group.index.name = 'age_group'
    return df_age_group

def strata_by_age_group(df, group_name, idx):
    '''stratify dataframe by label group index'''
    df_strata = df[df[group_name] == idx]
    return df_strata

def resample_data_by_group(df, n_samples):
    '''resample data by random replacement'''
    sample_group = resample(df, n_samples=n_samples, random_state=0, replace=True)
    return sample_group

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

def feature_scale_data(X):
    '''Feature scaled data based on standardization'''
    sc_X = StandardScaler()
    X_std = sc_X.fit_transform(X)
    return X_std
    
# Plot confusion matrix: accuracy, precision, recall and etc.
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

# Write report classification metrics summary report
def report_class_summary(model_name, y_act, y_pred):
    print ('Accuracy of ' + model_name + ' is %0.2f'% skm.accuracy_score(y_act, y_pred))
    print ('Precision of ' + model_name + ' is %0.2f'% skm.precision_score(y_act, y_pred))
    print ('Recall of ' + model_name + ' is %0.2f'% skm.recall_score(y_act, y_pred))
    print ('ROC score of ' + model_name + ' is %0.2f'% skm.roc_auc_score(y_act, y_pred))

# Compute confusion matrix:
def compute_confusion_matrix(y_act, y_pred):
    '''compute sklearn confusion matrix'''
    cm_model = skm.confusion_matrix(y_act, y_pred)
    return cm_model    

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

def model_results(model, mean_roc_auc, predictions, feature_importances):
    '''saves the model name, mean_roc_auc, predicted rate, and feature importances'''
    with open('model.txt', 'w') as file:
        file.write(str(model))
        feature_importances.to_csv('feat_importances.csv')
        predictions.to_csv('pred_results_best.csv', index=False)
    
# --- 3. Load the data --- #
if __name__ == '__main__':
# Define input CSVs:
    train_file = 'stroke_train.csv'
    test_file = 'stroke_test.csv'

# Define type of variables list:
#df_train.select_dtypes(include='object').columns
cat_vars = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

#df_train.select_dtypes(include='int64').columns
#df_train.select_dtypes(include='float64').columns
num_vars = ['hypertension', 'heart_disease', 'age', 'avg_glucose_level', 'bmi']
label_var = 'stroke'

# Define variables to drop
list_vars = 'id'

# Load data
df_train = load_file(train_file)
df_test = load_file(test_file)

# Check the metadata of dataframe:
df_train.info()

# Create a label dataframe:
df_label = df_train[['id', 'stroke']]

# Drop a column by index: poverty_rate
df_train = drop_column_by_index(df_train, label_var)

# join train set and label:
train_raw_df = join_data(df_train, df_label, key='id')

# --- 4. Perform data cleaning and quality check --- #
# Clean invalid data and duplicates: train and test set
clean_train_df = shuffle(clean_data(train_raw_df)).reset_index(drop=True)
clean_test_df = shuffle(clean_data(df_test)).reset_index(drop=True)

del (train_raw_df, df_test)

clean_train_df = drop_column_by_index(clean_train_df, list_vars)

# Compute missing value % on a dataframe:
missing_df_train = eda_missing_data(clean_train_df)
missing_df_test = eda_missing_data(clean_test_df)

# --- 5. Explore the data (EDA) --- # 
# Compute summary statistics
df_stat_num = eda_summary_stat_num(clean_train_df[num_vars])
df_stat_cat = eda_summary_stat_cat(clean_train_df)

# save row_id from test set:
df_test_id = clean_test_df['id']
clean_test_df = drop_column_by_index(clean_test_df, list_vars)

## Check 3: handling outliers using IQR #
################################################################################
## Compute IQR, LB, UB:        
#df_outliers = compute_outliers(df_stat_num)

# Correlation matrix plot # 
###########################
df_EDA = clean_train_df.copy()
df_EDA = drop_column_by_index(df_EDA, cat_vars)

# Plot correlation matrix
EDA_plot_correlation(df_EDA)

# Compute the correlation of each feature against poverty rate: order of magnitude
CM_summary = df_EDA.corr().sort_values(by=['stroke'], ascending=False)['stroke']
print(CM_summary)

# --- 6. Feature encode on categorical variables --- #
# Mapping ordinal and nominal features to integer:
smoking_status_map = {'never smoked':0, 'formerly smoked':1, 'smokes':2}
hypertension_map = {0:'No', 1:'Yes'}    
heart_disease_map = {0:'No', 1:'Yes'}

# Encode features into to mapping values: train set
clean_train_df['smoking_status'] = encode_categorical_feature(clean_train_df, 'smoking_status', smoking_status_map)
clean_train_df['hypertension'] = encode_categorical_feature(clean_train_df, 'hypertension', hypertension_map)
clean_train_df['heart_disease'] = encode_categorical_feature(clean_train_df, 'heart_disease', heart_disease_map)

# Encode features into to mapping values: test set
clean_test_df['smoking_status'] = encode_categorical_feature(clean_test_df, 'smoking_status', smoking_status_map)
clean_test_df['hypertension'] = encode_categorical_feature(clean_test_df, 'hypertension', hypertension_map)
clean_test_df['heart_disease'] = encode_categorical_feature(clean_test_df, 'heart_disease', heart_disease_map)

clean_train_df.isnull().sum()
clean_test_df.isnull().sum()

# --- 7. Feature imputation via univariate techniques --- #    
# Split data into input features and target variable #
X_train, y_train = split_dataframe(clean_train_df)
X_test = clean_test_df.copy()

del(clean_train_df, clean_test_df)

# check input features and target variable: train and test sets
print(X_train.head(), y_train.head())

# Imputation by mode, mean and median: train and test sets
indices_1 = range(8,9)
indices_2 = range(9,10)

# impute bmi and smoking_status by univariate methods: train and test sets
X_train.iloc[:, indices_1] = feature_imputer(X_train, 'NaN', 'median', indices_1)
X_train.iloc[:, indices_2] = feature_imputer(X_train, 'NaN', 'most_frequent', indices_2)

X_test.iloc[:, indices_1] = feature_imputer(X_test, 'NaN', 'median', indices_1)
X_test.iloc[:, indices_2] = feature_imputer(X_test, 'NaN', 'most_frequent', indices_2)

# concatenated imputed inputs and output label:
imputed_train_df = pd.concat([X_train, y_train], axis=1)
imputed_test_df = X_test.copy()

del(X_train, y_train, X_test)

# convert smoking_status back to original string:
inv_smoking_status_map = {key:value for value, key in smoking_status_map.items()}
imputed_train_df['smoking_status'] = encode_categorical_feature(imputed_train_df, 'smoking_status', inv_smoking_status_map)
imputed_test_df['smoking_status'] = encode_categorical_feature(imputed_test_df, 'smoking_status', inv_smoking_status_map)

# check any missing values on imputed df:
imputed_train_df.isnull().sum()
imputed_test_df.isnull().sum()

# check cleaned dataframe: data types
imputed_train_df.dtypes
imputed_test_df.dtypes

# Save feature_df for EDA portfolio:
#imputed_train_df.head()
#df_eda_stroke = imputed_train_df.copy()
#df_eda_stroke.to_csv('df_eda_stroke.csv', index=False)

# --- 8. Feature engineering: groupby categorical var ---

# convert data types for correct metadata: train and test sets
imputed_train_df['gender'] = convert_data_type(imputed_train_df, 'gender', 'category')
imputed_train_df['hypertension'] = convert_data_type(imputed_train_df, 'hypertension', 'category')
imputed_train_df['heart_disease'] = convert_data_type(imputed_train_df, 'heart_disease', 'category')
imputed_train_df['ever_married'] = convert_data_type(imputed_train_df, 'ever_married', 'category')
imputed_train_df['work_type'] = convert_data_type(imputed_train_df, 'work_type', 'category')
imputed_train_df['Residence_type'] = convert_data_type(imputed_train_df, 'Residence_type', 'category')
imputed_train_df['smoking_status'] = convert_data_type(imputed_train_df, 'smoking_status', 'category')

# --- 9. One-hot-encode on features --- # 
# Drop first dummy variable to avoid dummy variable trap on each converted feature!
num_vars2 = list(imputed_train_df.select_dtypes(include='float64').columns)
cat_vars2 = list(imputed_train_df.select_dtypes(include='category').columns)

feature_df = one_hot_encode_feature(imputed_train_df, cat_vars=cat_vars2, num_vars=num_vars2)
test_df = one_hot_encode_feature(imputed_test_df, cat_vars=cat_vars2, num_vars=num_vars2)

# List total number of encoded inputs and output:
feature_df.isnull().sum()
test_df.isnull().sum()

# Compute label: stroke
label_df = get_label_data(imputed_train_df, label_var)
#del(imputed_train_df, imputed_test_df)

# --- 10. Compute Proportion (%) of Stroke --- # 
df_feature_train = pd.concat([feature_df, label_df], axis=1)
df_stroke_pct = pd.DataFrame(df_feature_train.groupby('stroke')['age'].count())
df_stroke_pct.columns=['count']
df_stroke_pct['pct'] = (df_stroke_pct['count']/len(df_feature_train))*100

df_stroke_pct['pct'].plot(kind='pie', labels=['non-stroke','stroke'],
                         colors=['green','red'], autopct='%1.0f%%')

# --- 11. Resampling on non-stroke patients by non-stroke patients proportion --- # 
# --- SMOTE: oversample on minority class label --- #
X_train, X_val, y_train, y_val = train_test_split(feature_df, label_df, test_size=.1,
                                                    random_state=0, stratify=label_df)

sm = SMOTE(random_state=0, ratio='minority', n_jobs=-1)
X_train_resampled, y_train_resampled = sm.fit_sample(X_train, y_train)

pd.value_counts(y_train_resampled)

# --- 12. Feature seleciton: Feature Importance --- # 
# --- 13. Establish a baseline model --- # 
baseline = RandomForestClassifier(n_estimators=100, n_jobs=-1)
baseline.fit(X_train_resampled, y_train_resampled)

probs = baseline.predict_proba(X_val)
predictions = baseline.predict(X_val)

skm.roc_auc_score(y_val, probs[:,1]) # probabilities on 'stroke' cases only

test_probs = baseline.predict_proba(test_df)
results = pd.DataFrame({'id':df_test_id,
                        'stroke':test_probs[:,1]})
results.to_csv('prob_results_baseline.csv', index=False, index_label=None)

############################
# Part 3 - DEVELOP PHASE ###
############################
# --- 14. Create models --- # 
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

# --- 15. Cross-validate models --- # 
# 5-fold cross validation on models and measure MSE
# Model List to train: Order of Model Complexity
lr = LogisticRegression(tol=2.9078e-05, C=48, penalty='l1', solver='liblinear')
tree = DecisionTreeClassifier(max_depth=30, min_samples_split=4, min_samples_leaf=2,
                              max_features=9, random_state=0)
forest = RandomForestClassifier(n_estimators=87, max_depth=96, min_samples_split=4,
                                min_samples_leaf=2, random_state=0) 
xgb = XGBClassifier(n_estimators=79, max_depth=72, colsample_bytree=0.7970211308756587,
                    learning_rate=0.07492066671304065, random_state=0)
# List of classifiers:
models.extend([lr, tree, forest, xgb])

# cross-validate models, using MSE to evaluate and print the summaries
print('begin cross-validation')
for model in models:
    model_train(model, X_train_resampled, y_train_resampled, n_proc, mean_roc_auc, cv_std)
    model_summary(model, mean_roc_auc, cv_std)

# --- 16. Select the best model with lowest RMSE for your prediction model --- #
model = max(mean_roc_auc, key=mean_roc_auc.get)
print('\nBest model with the highest mean roc_auc:')
print(model)

# --- 17. Compute roc_auc score on "stroke cases only!" --- #
###############################################################################
score_model_roc_auc(lr, X_train_resampled,
                    y_train_resampled, X_val, y_val)

score_model_roc_auc(xgb, X_train_resampled,
                    y_train_resampled, X_val.values, y_val)

score_model_roc_auc(tree, X_train_resampled,
                    y_train_resampled, X_val, y_val)

score_model_roc_auc(forest, X_train_resampled,
                    y_train_resampled, X_val, y_val)

# --- 18. Model Evaluation: Confusion Matrix, Classification Metrics ---    
# Save cross-validated predictions:
lr.fit(X_train_resampled, y_train_resampled)
y_pred_lr = lr.predict(X_val)

tree.fit(X_train_resampled, y_train_resampled)
y_pred_tree = tree.predict(X_val)

forest.fit(X_train_resampled, y_train_resampled)
y_pred_forest = forest.predict(X_val)

xgb.fit(X_train_resampled, y_train_resampled)
y_pred_xgb = xgb.predict(X_val.values)

# Compute a series of confusion matrix by model:
cm_lr = compute_confusion_matrix(y_val, y_pred_lr)

cm_tree = compute_confusion_matrix(y_val, y_pred_tree)

cm_forest = compute_confusion_matrix(y_val, y_pred_forest)

cm_xgb = compute_confusion_matrix(y_val, y_pred_xgb)

# Define class labels for stroke:
class_labels = np.array(['non-stroke', 'stroke'], dtype=str)

#####################################################
# Confusion Matrix & Classification Metrics Summary #
#####################################################
# --- Logistic Regression ---#
# Plot a confusion matrix: 
plot_confusion_matrix(cm_lr, class_labels)
plt.title('Normalized Confusion Matrix: Logistic Regression')
plt.show()

# Report classification metrics summary:
report_class_summary('Logistic Regression', y_val, y_pred_lr)

# --- Decision Tree ---#
# Plot a confusion matrix: 
plot_confusion_matrix(cm_tree, class_labels)
plt.title('Normalized Confusion Matrix: Decision Tree')
plt.show()

# Report classification metrics summary:
report_class_summary('Decision Tree', y_val, y_pred_tree)

# --- Random Forest ---#
# Plot a confusion matrix: 
plot_confusion_matrix(cm_forest, class_labels)
plt.title('Normalized Confusion Matrix: Random Forest')
plt.show()

# Report classification metrics summary:
report_class_summary('Random Forest', y_val, y_pred_forest)

# --- XGBoost Classifier ---#
# Plot a confusion matrix: 
plot_confusion_matrix(cm_xgb, class_labels)
plt.title('Normalized Confusion Matrix: XGBoost')
plt.show()

# Report classification metrics summary:
report_class_summary('XGBoost', y_val, y_pred_xgb)

# --- 19. Model Evaluation: ROC-AUC Curve, Precision-Recall Curve --- 
clf_labels = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'XGBoost']
all_clf = [lr, tree, forest, xgb]

# plot a ROC-AUC curve
plt.figure(figsize=(8,8))
# ROC for each classifiers
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

# plot a Precision-Recall curve
# compute avg. precision score:
y_score_lr = lr.fit(X_train_resampled, y_train_resampled).predict_proba(X_val.values)[:,1]
avg_precision = skm.average_precision_score(y_val, y_score_lr)
print('Average P-R score of Logistic Regression: {0:0.2f}'.format(avg_precision))

# Plot a P-R curve:
precision, recall, _ =skm.precision_recall_curve(y_val, y_score_lr)
plt.step(recall, precision, color='navy', where='post', label='Precision-Recall Curve')
plt.title('P-R Curve: LogisiticRegression [test set] AP={0:0.2f}'.format(avg_precision))
plt.xlabel('Recall (Sensitivity)')
plt.ylabel('Precision (PPV)')
plt.ylim([0, 1.05])
plt.xlim([0, 1.0])
plt.legend(loc='lower right')
plt.show()

###########################
# Part 4 - DEPLOY PHASE ###
###########################
# --- 20. Automate the model pipeline --- #
# make predictions based on a test set
df_test_selected = test_df.copy()
df_pred_probs = lr.predict_proba(df_test_selected)

# make predictions dataframe:
results = pd.DataFrame({'id':df_test_id,
                        'stroke':df_pred_probs[:,1]})
results.to_csv('prob_results_lr.csv', index=False, index_label=None)

# --- 21. Deploy the solution --- #
#store feature importances
if hasattr(model, 'feature_importances_'):
    importances = model.feature_importances_
else:
# linear models don't have feature_importances_
    importances = [0]*len(X_train.columns)

# Create a feature importance dataframe and sort by importance:    
feature_importances = pd.DataFrame({'feature':X_train.columns,
                                        'importance':importances})
feature_importances.sort_values(by='importance', inplace=True, ascending=False)

# Set index to 'feature'
feature_importances.set_index('feature', inplace=True, drop=True)

# Create a bar plot:    
feature_importances[0:15].plot.bar(align='center')
plt.xticks(rotation=270, fontsize=9)
plt.title('feature importance plot: best trained classifier')
plt.show()
    
#Save model results as .csv file:
model_results(model, mean_roc_auc[model], results, feature_importances)