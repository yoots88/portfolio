# DSDJ Portfolio Practice - Salary Predictions Based on Job Descriptions #

###############################
# Part 1 - Define the Problem #
###############################

# Set Directory
import os
os.getcwd()

path = 'C:/Users/yoots/Desktop/P_Salary_Prediction'
#path = 'D:/P_Salary_Prediction'
os.chdir(path)

# Import all libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as sp
import sklearn.metrics as skm
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
#from statsmodels.formula.api import ols

# Authorization #
__author__ = "Taesun Yoo"
__email__ = "yoots.com"

#################################
# Part 2 - Discover the Problem #
#################################
# Write a group of funtions:
def load_data(file):
    '''Load input file CSVs to dataframe'''
    return pd.read_csv(file)

def combine_data(df1, df2, key=None, left_index=False, right_index=False):
    '''Performs inner join on two dataframes and return records'''
    return pd.merge(left=df1, right=df2, how='inner', on=key,
                       left_index=left_index, right_index=right_index)

def clean_data(raw_df):
    '''Remove rows with duplicate jobId and invalid records'''
    cleaned_df = raw_df.drop_duplicates(subset='jobId')
    cleaned_df = cleaned_df[cleaned_df.salary>0]
    return cleaned_df

def left_join_data(cleaned_df, avg_groupby_cat_df, key=None, left_index=False, right_index=False):
    '''Performs left join from cleaned dataframe to average groupby dataframe'''
    return pd.merge(left=cleaned_df, right=avg_groupby_cat_df, how='left',
                   left_index=left_index, right_index=right_index)

def EDA_pivot_table(cleaned_df, cat_var, num_var):
    '''Creates a pivot table based on categorical var and average numerical var'''
    pivot_cat_df = cleaned_df.pivot_table(index=cat_var, values=num_var, aggfunc=np.mean)
    pivot_cat_df.reset_index(level=0, inplace=True)
    pivot_cat_df.rename(columns={cat_var:cat_var, num_var:"avg" + "_" + num_var + "_" + cat_var}, inplace=True)
    return pivot_cat_df

def drop_cat_feature_df(df, list_vars=None):
    '''Drop no-value added categorical features'''
    final_df = df.drop(list_vars, axis=1)
    return final_df

def one_hot_encode_feature_df(df, cat_vars=None, num_vars=None):
    '''performs one-hot encoding on all categorical variables and combine
       results with numerical variables'''
    cat_df = pd.get_dummies(df[cat_vars], drop_first=True)
    num_df = df[num_vars].apply(pd.to_numeric)
    return pd.concat([cat_df, num_df], axis=1)

def get_label_df(df, label):
    '''Returns label from dataframe'''
    return df[label]

def feat_selection_df(df, model, feat_name, Est_Coef, file_name):
    '''Creates L1 feature selected dataframe'''
    df_L1_select = pd.DataFrame({feat_name:df.columns, Est_Coef:model.coef_})[[feat_name, Est_Coef]]
    df_L1_select = df_L1_select.sort_values(by=Est_Coef, ascending=False)
    df_L1_select.to_excel(file_name)

def model_tuning_param(model, feature_df, label_df, param_dist, n_iter):
    '''Performs RandomizedSearchCV to tune model parameters'''
    random_search = RandomizedSearchCV(model, param_dist, n_iter, cv=2)
    random_search.fit(feature_df, label_df)
    return random_search

def print_best_params(random_search, param_1=None, param_2=None, param_3=None, param_4="None"):
    '''Print the best model parameter'''
    print("Best " + param_1 + ":", random_search.best_estimator_.get_params()[param_1])
    print("Best " + param_1 + ":", random_search.best_estimator_.get_params()[param_2])
    print("Best " + param_1 + ":", random_search.best_estimator_.get_params()[param_3])
    print("Best " + param_1 + ":", random_search.best_estimator_.get_params()[param_4])

def model_train(model, feature_df, label_df, n_proc, mean_mse, cv_std):
    '''Train a model and output mean MSE and CV Std.Dev MSE'''
    neg_mse = cross_val_score(model, feature_df, label_df, n_jobs=n_proc, cv=2,
                              scoring='neg_mean_squared_error')
    mean_mse[model] = -1.0*np.mean(neg_mse)
    cv_std[model] = np.std(neg_mse)
    
def model_summary(model, mean_mse, cv_std):
    '''Print out the model perforamnces'''
    print('\nModel:\n', model)
    print('Average MSE:\n', mean_mse[model])
    print('Std. Dev during CV:\n', cv_std[model])
    
def model_results(model, mean_mse, predictions, feature_importances):
    '''Saves the model name, mean_mse, predicted salaries, and feature importances'''
    with open('model.txt', 'w') as file:
        file.write(str(model))
        feature_importances.to_csv('feat_importances.csv')
        np.savetxt('final_pred.csv', predictions, delimiter=',')
        
if __name__ == '__main__':
    #define input CSVs
    train_feature_file='C:/Users/yoots/Desktop/P_Salary_Prediction/train_features.csv'
    train_label_file='C:/Users/yoots/Desktop/P_Salary_Prediction/train_salaries.csv'
    test_feature_file='C:/Users/yoots/Desktop/P_Salary_Prediction/test_features.csv'
    
    #define variable lists
#    categorical_vars = ['jobId', 'companyId',  'jobType', 'degree', 'major', 'industry']
    categorical_vars = ['companyId',  'jobType', 'degree', 'major']
    numerical_vars = ['yearsExperience', 'milesFromMetropolis', 'avg_salary_industry', 'avg_yearsExperience_industry', 
                      'avg_yearsExperience_jobType','avg_milesFromMetropolis_industry','avg_milesFromMetropolis_jobType']
    label_var = 'salary'
    
    #define variables to drop
    list_vars = ['jobId', 'industry']
    
    #load data
    print("data loading")
    train_df = load_data(train_feature_file)
    label_df = load_data(train_label_file)
    test_df = load_data(test_feature_file)
    
    del(train_feature_file, train_label_file, test_feature_file)
    
    # combine train set
    train_raw_df = combine_data(train_df, label_df, key='jobId')
    
    #clean, shuffle, and reindex train data -- shuffling improves c-v accuracy
    cleaned_df = shuffle(clean_data(train_raw_df)).reset_index(drop=True)
    
#---- 5 Engineer features ----#
# ensure data is ready for modeling
# create any new features needed to potentially enhance model
#---- Compute Pivot Table ----#
    pivot_salary_industry_df = EDA_pivot_table(cleaned_df, 'industry', 'salary')
    pivot_yearsExp_industry_df = EDA_pivot_table(cleaned_df, 'industry', 'yearsExperience')
    pivot_yearsExp_jobType_df = EDA_pivot_table(cleaned_df, 'jobType', 'yearsExperience')
    pivot_dist_industry_df = EDA_pivot_table(cleaned_df, 'industry', 'milesFromMetropolis')
    pivot_dist_jobType_df = EDA_pivot_table(cleaned_df, 'jobType', 'milesFromMetropolis')
        
    #perform left joins on avg_salary and train/test set:
    joined_train_df = left_join_data(cleaned_df, pivot_salary_industry_df, key='industry')    
    joined_train_df = left_join_data(joined_train_df, pivot_yearsExp_industry_df, key='industry')    
    joined_train_df = left_join_data(joined_train_df, pivot_yearsExp_jobType_df, key='jobType')    
    joined_train_df = left_join_data(joined_train_df, pivot_dist_industry_df, key='industry')    
    joined_train_df = left_join_data(joined_train_df, pivot_dist_jobType_df, key='jobType')    
    
    joined_test_df = left_join_data(test_df, pivot_salary_industry_df, key='industry')    
    joined_test_df = left_join_data(joined_test_df, pivot_yearsExp_industry_df, key='industry')    
    joined_test_df = left_join_data(joined_test_df, pivot_yearsExp_jobType_df, key='jobType')    
    joined_test_df = left_join_data(joined_test_df, pivot_dist_industry_df, key='industry')    
    joined_test_df = left_join_data(joined_test_df, pivot_dist_jobType_df, key='jobType')    
    
    del(pivot_salary_industry_df, pivot_yearsExp_industry_df, pivot_yearsExp_jobType_df,
    pivot_dist_industry_df, pivot_dist_jobType_df)    

    # drop useless categorical features on train and test sets
    print("drop useless cat features")
    train_df = drop_cat_feature_df(joined_train_df, list_vars=list_vars)
    test_df = drop_cat_feature_df(joined_test_df, list_vars=list_vars)

    # one-hot encode categorical data on train and test sets
    print("encoding data")
    train_df = one_hot_encode_feature_df(train_df, cat_vars=categorical_vars, 
                                         num_vars=numerical_vars)
    test_df = one_hot_encode_feature_df(test_df, cat_vars=categorical_vars, 
                                         num_vars=numerical_vars)

    del(joined_train_df, joined_test_df)
    
    # get label df
    label_df = get_label_df(cleaned_df, label_var)

    del(cleaned_df)
        
    # ----- Feature Selection: L1/L2 Regularization ----- #
    # fit L1 to filter out non-zero coefficients
#    lr_L1 = Lasso(alpha=0.01)
#    lr_L1.fit(train_df, label_df)

    # Print estimated intercept coefficients:
#    print('Est. coefficient: {}'.format(lr_L1.intercept_))

    #Save L1 feature selection results:
#    feat_selection_df(train_df, lr_L1, 'features', 'EstCoef', 'L1_feat_selection_df.xlsx')    
    
    #Create L1 feature selection index:
    idx_L1 = np.r_[62:84,86]

    #Filtered train_df and test_df
    train_L1_df = train_df[train_df.columns[idx_L1]]
    test_L1_df = test_df[test_df.columns[idx_L1]]

#---- 3 Establsih a baseline ----#
# Select a reasonable metric (i.e., MSE, RMSE)
# Create an extremely simple model and measure the efficacy
#    lr = LinearRegression()
#    baseline_cv = cross_val_score(lr, train_L1_df, label_df, 
#                                  cv=5, scoring='neg_mean_squared_error')
#    baseline_mse = -1.0*np.mean(baseline_cv)
#    print('Baseline MSE: ' + str(baseline_mse))

## Test: PolynomialRegression
    pr_std = make_pipeline(StandardScaler(), PolynomialFeatures(degree=2), LinearRegression())
    pr_std_cv = cross_val_score(pr_std, train_L1_df, label_df, 
                                  cv=5, scoring='neg_mean_squared_error')
    pr_std_mse = -1.0*np.mean(pr_std_cv)
    print('PolynomialReg. MSE: ' + str(pr_std_mse))

## Test: GradientBoosting
#    gbr = GradientBoostingRegressor(n_estimators=40, max_depth=5, loss='ls', verbose=5)
#    gbr_cv = cross_val_score(gbr, train_L1_df, label_df, 
#                                  cv=5, scoring='neg_mean_squared_error')
#    gbr_mse = -1.0*np.mean(gbr_cv)
#    print('GradientBoostingReg. MSE: ' + str(gbr_mse))

#---- 4 Hypothesize solution ----#
# Brainstorm 3 regression models that may improve the results over the baseline model

# Your metric will be MSE and goals are following: repeat steps 5-8
# <360 for entry-level data science roles
# <320 for senior data science roles

#---- 6 Create models ----#
# Create and tune the models that you brainstormed during part 2

    # initialize model list and dictionaries
    models = []
    mean_mse = {}
    cv_std = {}
    res = {}
    
    # define num of cores to run in parallel
    n_proc = 1
    # shared model parameters
    verbose_lv = 5

###############################################################################        
#    #model tuning with RandomizedSearchCV: RandomForest
#    rf = RandomForestRegressor()   
#    n_iter = 1
#    param_dist_rf = {'n_estimators':sp.randint(10,50), 
#                  'max_depth':sp.randint(1,10),
#                  'min_samples_split':sp.randint(10,60),
#                  'max_features':sp.randint(0,23)}
#    random_search_rf = model_param_tuning(rf, train_L1_df, label_df, param_dist_rf, n_iter)    
#    
#    # print the best model parameters: RandomForest    
#    param_1 = 'n_estimators' 
#    param_2 = 'max_depth'
#    param_3 = 'min_samples_split'
#    param_4 = 'max_features'
#    
#    print_best_params(random_search_rf, param_1, param_2, param_3, param_4)
###############################################################################    
##model tuning with RandomizedSearchCV: GradientBoosting
#    gbr = GradientBoostingRegressor()
#    n_iter = 1
#    param_dist_gbr = {'n_estimators':sp.randint(10,40), 
#                  'max_depth':sp.randint(1,20),
#                  'loss':['ls']}
#    random_search_gbr = model_param_tuning(gbr, train_L1_df, label_df, param_dist_gbr, n_iter)    
#    
#    # print the best model parameters: GradientBoosting    
#    param_1 = 'n_estimators' 
#    param_2 = 'max_depth'
#    param_3 = 'loss'
#    
#    print_best_params(random_search_gbr, param_1, param_2, param_3)        
###############################################################################    
    #model tuning with RandomizedSearchCV: XGBRegressor
    xgb = XGBRegressor()   
    n_iter = 1
    param_dist_xgb = {'n_estimators':sp.randint(10,40), 
                  'max_depth':sp.randint(1,20),
                  'learning_rate':np.random.uniform(0,1,10)}
    random_search_xgb = model_tuning_param(xgb, train_L1_df, label_df, param_dist_xgb, n_iter)    
    
    # print the best model parameters: XGBRegressor    
    param_1 = 'n_estimators' 
    param_2 = 'max_depth'
    param_3 = 'learning_rate'
    
    print_best_params(random_search_xgb, param_1, param_2, param_3)    
###############################################################################    
# Model List to Train: Order of Model Complexity
    lr_L1 = Lasso(alpha=0.01)
    lr_L2 = Ridge(alpha=0.01)
#    lr_std_pca = make_pipeline(StandardScaler(), PCA(), LinearRegression())
#    pr_std = make_pipeline(StandardScaler(), PolynomialFeatures(degree=2), LinearRegression())  
    rf = RandomForestRegressor(n_estimators=43, n_jobs=n_proc, max_depth=5,
                               min_samples_split=16, max_features=16, verbose=verbose_lv)   
    xgb = XGBRegressor(n_estimators=33, max_depth=5, learning_rate=0.831381818143282) 
#    gbr = GradientBoostingRegressor(n_estimators=40, max_depth=5, loss='ls', verbose=verbose_lv)
 
    models.extend([lr_L1, lr_L2, rf, xgb])    

#---- 7 Test models ----#
# perform 5-fold corss validation and measure MSE
    
    #cross-validate models, using MSE to evaluate and print the summaries
    print("begin cross-validation")
    for model in models:
        model_train(model, train_df, label_df, n_proc, mean_mse, cv_std)
        model_summary(model, mean_mse, cv_std)

#---- 8 Select best model ----#
# select the model with the lowest MSE as your "production model"

    #choose model with the Lowest MSE
    model = min(mean_mse, key=mean_mse.get)
    print('\nBest model with the lowest MSE:')
    print(model)

#---- 9 Automate pipeline ----#
# wrtie script that trains model on entire training set, saves model to disk,
# and scores the "test" dataset
    
    #train model on entire dataset
    model.fit(train_L1_df, label_df)
    
    #make predictions based on test set
    pred = model.predict(test_L1_df)
    
    #store feature importances
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        #lineear models don't have feature_importances
        importances = [0]*len(train_L1_df.columns)
    
    feature_importances = pd.DataFrame({'feature':train_L1_df.columns, 
                                        'importance':importances})
    feature_importances.sort_values(by='importance', ascending=False,
                                    inplace=True)
    
    #set index to 'feature'
    feature_importances.set_index('feature', inplace=True, drop=True)
    
    #create a plot
    feature_importances[0:20].plot.bar(align='center')
    plt.xticks(rotation=45, fontsize=7)
    
#---- 10 Deploy solution ----#
# save your prediction to a csv file or optionally save them as a table in a SQL database
# additionally, you want to save a visualization and summarize your prediction and feature importances
# These summaries will provide insights to the business stakeholders
    model_results(model, mean_mse[model], pred, feature_importances)