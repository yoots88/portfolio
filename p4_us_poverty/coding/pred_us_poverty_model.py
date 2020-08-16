# Import and set directory:
import os
os.getcwd()

# Set path:
path = 'D:/TY Workspace/3. Data Science Portfolio/1. Completed Portfolios/P4_US_Poverty_Prediction/2_Development'
os.chdir(path)

# Import all libraries #
import sys
import numpy as np
import pandas as pd
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
    '''remove rows that contain invalid data or duplicate IDs'''
    clean_df = raw_df.drop_duplicates(subset='row_id')
    clean_df = clean_df[clean_df['health__homicides_per_100k'] > 0]
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


def model_train(model, feature_df, label_df, n_proc, mean_rmse, cv_std):
    '''Train a model and output mean RMSE and CV Std.Dev RMSE''' 
    neg_mse = cross_val_score(model, feature_df, label_df, n_jobs=n_proc, cv=5,
                              scoring='neg_mean_squared_error')
    rmse = np.sqrt(-1 * neg_mse)
    mean_rmse[model] = np.mean(rmse)
    cv_std[model] = np.std(rmse)


def model_summary(model, mean_rmse, cv_std):
    '''Print out the model perforamnces'''
    print('\nModel:\n', model)
    print('Average RMSE:\n', mean_rmse[model])
    print('Std. Dev during CV:\n', cv_std[model]) 

    
def model_results(model, mean_rmse, predictions, feature_importances):
    '''Saves the model name, mean_rmse, predicted rate, and feature importances'''
    with open('model.txt', 'w') as file:
        file.write(str(model))
        feature_importances.to_csv('feat_importances.csv')
#        predictions.to_csv('final_predictions.csv', index=False)


def plot_reg_compare(y_train_pred, y_train_act):
    '''Plot a scatter plot to compare predicted vs actual label'''
    plt.scatter(y_train_pred, y_train_act, c='blue', 
                edgecolor='w', marker='o', label='train set')
    plt.xlabel('Predicted poverty rate')
    plt.ylabel('Actual poverty rate')
    plt.legend(loc='upper left')


def plot_reg_residual(y_train_pred, y_train_act):
    '''Plot a scatter plot to visualize residual from predicted vs. actual label'''
    plt.scatter(y_train_pred, (y_train_pred - y_train_act), c='blue',
                edgecolor='w', marker='o', label='train set')
    plt.hlines(y=0, xmin=0, xmax=max(y_train_act), colors='k', lw=3)
    plt.xlim([0, max(y_train_act)])
    plt.xlabel('Predicted poverty rate')
    plt.ylabel('Residual poverty rate')
    plt.legend(loc='upper left')    

# --- 2. Load the data --- #
# define input CSVs:
if __name__ == '__main__':
    train_file ='poverty_train.csv'
    test_file ='poverty_test.csv'

# Load data
df_train = load_file(train_file)
df_test = load_file(test_file)
del(train_file, test_file)

# define variables:
var_label = 'poverty_rate'
var_id = 'row_id'
vars_cat = list(df_train.select_dtypes(include='O').columns)
vars_num_disc = list(df_train.select_dtypes(include='int64').columns)
vars_num_cont = list(df_train.select_dtypes(include='float64').columns)

# concatenate the two lists:
vars_num = vars_num_disc + vars_num_cont

list_unwanted = {'row_id','poverty_rate'}

vars_num = [item for item in vars_num if item not in list_unwanted]

del(vars_num_disc, vars_num_cont)

# check data types on dataframes:
df_train.info()

# --- 3. ETL - metadata format --- #

# --- 4. ETL - merging/subsetting data --- #
# create a dataframe with label:
df_label = df_train[['row_id','poverty_rate']]

# drop a specified column:
df_train = drop_column(df_train, var_label)

# merge on a train set and label:
df_train_raw = join_data(df_train, df_label, 'inner', 'row_id')

# --- 5. ETL - cleaning data --- #
# clean data:
df_train_clean = shuffle(clean_data(df_train_raw)).reset_index(drop=True)
df_test_clean = shuffle(clean_data(df_test)).reset_index(drop=True)
del(df_test, df_train, df_train_raw)

# create a dataframe: test set 'id'
df_test_id = df_test_clean['row_id']

# --- 6. Feature Encoding --- #
# Define manual mapping for area__rucc:
urban_deg_map ={'Nonmetro - Completely rural or less than 2,500 urban population, not adjacent to a metro area': 1,
        'Nonmetro - Completely rural or less than 2,500 urban population, adjacent to a metro area': 1,
        'Nonmetro - Urban population of 2,500 to 19,999, not adjacent to a metro area': 2,
        'Nonmetro - Urban population of 2,500 to 19,999, adjacent to a metro area': 2,
        'Nonmetro - Urban population of 20,000 or more, not adjacent to a metro area': 3,
        'Nonmetro - Urban population of 20,000 or more, adjacent to a metro area': 3,
        'Metro - Counties in metro areas of fewer than 250,000 population': 4,
        'Metro - Counties in metro areas of 250,000 to 1 million population': 5,
        'Metro - Counties in metro areas of 1 million population or more': 6}

# Define manual mapping for area__urban_influence:
urban_size_map ={'Noncore adjacent to a small metro and does not contain a town of at least 2,500 residents': 1,
               'Noncore adjacent to a small metro with town of at least 2,500 residents': 2,
               'Noncore not adjacent to a metro/micro area and does not contain a town of at least 2,500 residents': 3,
               'Noncore not adjacent to a metro/micro area and contains a town of 2,500  or more residents': 4,
               'Noncore adjacent to micro area and contains a town of 2,500-19,999 residents': 5,
               'Noncore adjacent to a large metro area': 6,
               'Micropolitan not adjacent to a metro area': 7,
               'Micropolitan adjacent to a small metro area': 8,
               'Micropolitan adjacent to a large metro area': 9,
               'Small-in a metro area with fewer than 1 million residents': 10,
               'Large-in a metro area with at least 1 million residents or more': 11}

# Define manual mapping for yr:
yr_map ={'a':'year 1', 'b':'year 2'}

# encode features on a train set:
df_train_clean['area__rucc'] = encode_categorical_feature(df_train_clean,'area__rucc',urban_deg_map)
df_train_clean['area__urban_influence'] = encode_categorical_feature(df_train_clean,'area__urban_influence',urban_size_map)
df_train_clean['yr'] = encode_categorical_feature(df_train_clean,'yr',yr_map)

# encode features on a test set:
df_test_clean['area__rucc'] = encode_categorical_feature(df_test_clean,'area__rucc',urban_deg_map)
df_test_clean['area__urban_influence'] = encode_categorical_feature(df_test_clean,'area__urban_influence',urban_size_map)
df_test_clean['yr'] = encode_categorical_feature(df_test_clean,'yr',yr_map)

del (urban_deg_map, urban_size_map, yr_map)

# --- 7. Feature Imputation --- #
# Compute missing value %: before replacement
df_missing_pre = EDA_missing_data(df_train_clean)
df_missing_pre

# feature imputation:
feature_replacement(df_train_clean)
feature_replacement(df_test_clean)

# Compute missing value %: after replacement
df_missing_post = EDA_missing_data(df_train_clean)
df_missing_post

del(df_missing_pre, df_missing_post)

# --- 8. Feature Engineering --- #
# average numeric by categorical variables:
df_avg_poverty_by_econ = avg_groupby_data(df_train_clean, 'poverty_rate', 'econ__economic_typology', 'avg_poverty_by_econ_type')
df_avg_pct_no_diploma_by_econ = avg_groupby_data(df_train_clean, 'demo__pct_adults_less_than_a_high_school_diploma', 'econ__economic_typology', 'avg_pct_no_diploma_by_econ_type')
df_avg_pct_unemployment_by_econ = avg_groupby_data(df_train_clean, 'econ__pct_unemployment', 'econ__economic_typology', 'avg_pct_unemployment_by_econ_type')

# perform left joins on avg grouped dataframes:
df_train_clean  = join_data(df_train_clean, df_avg_poverty_by_econ, 'left', key='econ__economic_typology')
df_train_clean  = join_data(df_train_clean, df_avg_pct_no_diploma_by_econ, 'left', key='econ__economic_typology')
df_train_clean  = join_data(df_train_clean, df_avg_pct_unemployment_by_econ, 'left', key='econ__economic_typology')

df_test_clean = join_data(df_test_clean, df_avg_poverty_by_econ, 'left', key='econ__economic_typology')
df_test_clean = join_data(df_test_clean, df_avg_pct_no_diploma_by_econ, 'left', key='econ__economic_typology')
df_test_clean = join_data(df_test_clean, df_avg_pct_unemployment_by_econ, 'left', key='econ__economic_typology')

del(df_avg_poverty_by_econ, df_avg_pct_no_diploma_by_econ, df_avg_pct_unemployment_by_econ)

# --- 9. Exploratory Data Analysis --- #
# perform numerical stat:
df_stat_num = EDA_summary_stat_num(df_train_clean[vars_num])
df_stat_num

# convert data type as category:
convert_dt_as_category(df_train_clean)
convert_dt_as_category(df_test_clean)

# perform summary statistics: categorical
df_stat_cat = EDA_summary_stat_cat(df_train_clean[vars_cat])
df_stat_cat

# --- 10. Prepare Training Data --- #
# Drop first dummy variable on each nominal feature to avoid dummy variable trap:
df_feature_enc = one_hot_encode_feature(df_train_clean, cat_vars=vars_cat, num_vars=vars_num)
df_test_enc = one_hot_encode_feature(df_test_clean, cat_vars=vars_cat, num_vars=vars_num)

# create a label:
label_df = df_train_clean['poverty_rate']

# --- 11. variable inflation factor (VIF): deal with multi-colinearity --- # 
# get variables for computing VIF and add intercept term
X_vif = df_feature_enc.copy()
X_vif['intercept'] = 1

# Compute and view VIF
df_vif_view = pd.DataFrame()
df_vif_view['feature_name'] = X_vif.columns
df_vif_view['score_vif'] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]

# check results where VIF is less than 10:
print(df_vif_view[df_vif_view['score_vif']<10])

# Create a column list: where VIF is less than 10
list_vif_cols = list(df_vif_view['feature_name'][df_vif_view['score_vif']<10])
list_vif_cols.remove('intercept')

# Dataframe selected with VIF:
df_feat_vif_select = X_vif[list_vif_cols]
del (X_vif, list_vif_cols)

# --- 12. Feature Selection: L1 regularization --- # 
# fit L1 to filter out non-zero coefficients
lr_L1 = Lasso(alpha=0.01)
lr_L1.fit(df_feature_enc, label_df)

# Print estimated intercept coefficients:
print('Est. coefficient: {}'.format(lr_L1.intercept_))

# Save L1 feature selection results:
df_feat_L1_select = feature_selection(df_feature_enc, lr_L1, 'feature', 'EstCoef', 'df_feat_L1_selected.xlsx')   

# Create a column list: where L1 Estimated Coefficient is NOT ZERO.
list_L1_cols = list(df_feat_L1_select['feature'][df_feat_L1_select['EstCoef'] != 0])

# Dataframe filtered with L1 selected features:
df_train_L1 = df_feature_enc[list_L1_cols]
df_test_L1 = df_test_enc[list_L1_cols]

# --- 13. split data into train and test set ---
X_train, X_val, y_train, y_val = train_test_split(df_feature_enc, label_df, 
                                                  test_size=1/4, random_state=0)

# check the split:
y_train.count()
y_val.count()

# --- 14. Baseline model --- # 
# Create a baseline model:
#baseline = make_pipeline(StandardScaler(), PolynomialFeatures(degree=2), LinearRegression())
baseline = LinearRegression()
baseline_cv = cross_val_score(baseline, X_train, y_train, 
                                  cv=5, scoring='neg_mean_squared_error')
baseline_rmse = np.sqrt(-1 * baseline_cv)
baseline_mean_rmse = np.mean(baseline_rmse)
print('Baseline RMSE: ' + str(baseline_mean_rmse))

############################
# Part 3 - DEVELOP PHASE ###
############################
# --- 15. Create models --- # 
# initialize model list and dicts
models = []
mean_rmse = {}
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

# Hyper-parameters tuning: GradientBoosting
gbr = GradientBoostingRegressor()
n_iter = 10
param_dist_gbr = {'n_estimators':sp.randint(10,40), 
                  'max_depth':sp.randint(1,20),
                  'loss':['ls']}

random_search_gbr = model_tuning_param(gbr, X_train, y_train, param_dist_gbr, n_iter)    
    
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
rf = RandomForestRegressor(n_estimators=47, n_jobs=n_proc, max_depth=9,
                               min_samples_split=40, verbose=verbose_lv)   
gbr = GradientBoostingRegressor(n_estimators=34, max_depth=8, loss='ls', verbose=verbose_lv)
xgb = XGBRegressor(n_estimators=31, max_depth=4, learning_rate=0.338114) 

# a list of regressors:
models.extend([lr_L1, lr_L2, rf, gbr, xgb])    

# cross-validate models, using MSE to evaluate and print the summaries
print("begin cross-validation")
for model in models:
    model_train(model, X_train, y_train, n_proc, mean_rmse, cv_std)
    model_summary(model, mean_rmse, cv_std)

# --- 17. Select the best model with lowest RMSE for your prediction model --- #
model = min(mean_rmse, key=mean_rmse.get)
print('\nBest model with the lowest RMSE:')
print(model)

# --- 18. Model Evaluation: Scatter Plot --- #
# Prepare predicted and original poverty rate
y_train_act = y_train.copy()

# re-train a model with best model:
model.fit(X_train, y_train,)
y_train_pred = model.predict(X_train)
        
# Plot a comparison scatter plot: predicted vs. actual
plt.figure(figsize=(14,7))
plt.subplots_adjust(wspace=0.2)
plt.subplot(1,2,1)
plot_reg_compare(y_train_pred, y_train_act)
plt.title('Poverty rate of predicted vs. actual: best trained model')

# Plot a residual scatter plot: predicted vs. actual
plt.subplot(1,2,2)
plot_reg_residual(y_train_pred, y_train_act)
plt.title('Poverty rate of predicted vs. residual: best trained model')
plt.show()

###########################
# Part 4 - DEPLOY PHASE ###
###########################
# --- 19. Automate the model pipeline --- #
# make predictions on a test set
df_pred = model.predict(df_test_enc)

# make predictions as a dataframe:
results = pd.DataFrame({'id':df_test_id,
                        'provery_rate':df_pred})
results.to_csv('pred_results_best.csv', index=False, index_label=None)

# --- 20. Deploy the solution --- #
#store feature importances
if hasattr(model, 'feature_importances_'):
    importances = model.feature_importances_
else:
#linear models don't have feature_importances
    importances = [0]*len(df_feature_enc.columns)
    
feature_importances = pd.DataFrame({'feature':df_feature_enc.columns, 
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
model_results(model, mean_rmse[model], df_pred, feature_importances)