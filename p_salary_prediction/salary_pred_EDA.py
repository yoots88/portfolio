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
import sklearn.metrics as skm
from sklearn.utils import shuffle

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

def left_join_data(cleaned_df, pivot_cat_df, key=None, left_index=False, right_index=False):
    '''Performs left join from cleaned dataframe to pivot avg. mean dataframe'''
    return pd.merge(left=cleaned_df, right=pivot_cat_df, how='left',
                   left_index=left_index, right_index=right_index)

def EDA_missing_data(cleaned_df):
    '''Performs missing % on each column '''
    missing_df = cleaned_df.isnull().sum()
    missing_df = pd.DataFrame(missing_df, columns=['count'])
    missing_df['pct'] = missing_df['count']/len(cleaned_df)
    missing_df = missing_df.sort_values(by='pct', ascending=False)
    return missing_df

def EDA_numerical_data(cleaned_df):
    '''Computes summary statistics on numerical data'''
    summary_df_num = cleaned_df.describe(include='all').T
    summary_df_num = pd.DataFrame(summary_df_num)[['count', 'std', 'min', 'mean', 'max', '25%', '50%', '75%']]
    return summary_df_num

def EDA_categorical_data(cleaned_df):
    '''Computes summary statitics on categorical data'''
    summary_df_cat = cleaned_df.describe(include=['O'])
    summary_df_cat = pd.DataFrame(summary_df_cat)
    return summary_df_cat
    
def EDA_pivot_table(cleaned_df, cat_var, num_var):
    '''Creates a pivot table based on categorical var and average numerical var'''
    pivot_cat_df = cleaned_df.pivot_table(index=cat_var, values=num_var, aggfunc=np.mean)
    pivot_cat_df.reset_index(level=0, inplace=True)
    pivot_cat_df.rename(columns={cat_var:cat_var, num_var:"avg" + "_" + num_var + "_" + cat_var}, inplace=True)
    return pivot_cat_df

def EDA_plot_hist(cleaned_df, var_name, color):
    '''Print skewness and plot the histogram'''
    print("Skewness is:", cleaned_df[var_name].skew())
    plt.figure()
    plt.hist(cleaned_df[var_name], color=color)
    plt.xlabel(var_name)
    plt.ylabel('frequency counts')
    plt.title('Histogram of ' + var_name)
    
def EDA_plot_freq_chart(df, cat_var):
    cat_var_count = df[cat_var].value_counts()
    sns.barplot(cat_var_count.index, cat_var_count.values, alpha=0.9)
    plt.title('Frequency Counts of '+ cat_var)
    plt.ylabel('Counts')
    plt.xlabel(cat_var, fontsize=12)
    plt.show()

def EDA_plot_bar(cleaned_df, cat_var, num_var, color):
    '''Plots the bar chart'''
    cleaned_df.plot.bar(color=color)
    plt.xlabel(cat_var)
    plt.ylabel('Avg. ' + num_var)
    plt.xticks(rotation=0)
    plt.show()    

def EDA_plot_box_whisker(df, num_var, cat_var, hue=None):
    df.sort_values(by=[num_var], inplace=True)
    plt.figure()
    sns.set(style='whitegrid')
    sns.boxplot(cat_var, num_var, hue, df)
    plt.title('Box Plot of '+ num_var + ' by '+ cat_var)

def EDA_convert_object_to_cat(df):
    for col in df.columns:
        if df[col].dtype.name == "object":
            df[col] = df[col].astype('category')

def EDA_encode_cat_var(df, col):
    '''encode the categorical variables using avg. salary for each category '''
    cat_dict={}
    cats = df[col].cat.categories.tolist()
    for cat in cats:
        cat_dict[cat] = df[df[col] == cat]['salary'].mean()
    df[col] = df[col].map(cat_dict)

def EDA_plot_corr_matrix(df, features, label):
    plt.figure(figsize=(12,10))
    sns.heatmap(df[features + label].corr(), cmap=sns.diverging_palette(220, 10, as_cmap=True), annot=True)
    plt.xticks(rotation=45)
    plt.show()

if __name__ == '__main__':
    #define input CSVs
    train_feature_file='C:/Users/yoots/Desktop/P_Salary_Prediction/train_features.csv'
    train_label_file='C:/Users/yoots/Desktop/P_Salary_Prediction/train_salaries.csv'
    test_feature_file='C:/Users/yoots/Desktop/P_Salary_Prediction/test_features.csv'
    
    #define variable lists
    categorical_vars = ['companyId',  'jobType', 'degree', 'major', 'industry']
    numerical_vars = ['yearsExperience', 'milesFromMetropolis'] 

    label_var = ['salary']
    
    #define variables to drop
    list_vars = ['jobId']
    
    #load data
    print("data loading")
    train_df = load_data(train_feature_file)
    label_df = load_data(train_label_file)
    test_df = load_data(test_feature_file)
    
    del(train_feature_file, train_label_file, test_feature_file)
    
    # combine train set: train features and train labels (salaries)
    train_raw_df = combine_data(train_df, label_df, key='jobId')
    
    #clean, shuffle, and reindex train data -- shuffling improves c-v accuracy
    cleaned_df = shuffle(clean_data(train_raw_df)).reset_index(drop=True)

################################################
# Part 3 - Exploratory Data Analysis: Insights #
################################################
    
#---- Compute % of Missing Data ----#
    missing_df = EDA_missing_data(cleaned_df)
    print(missing_df)
    
#---- Compute Summary Statistics: numerical data ----#
    summary_df_num = EDA_numerical_data(cleaned_df[numerical_vars])
    print(summary_df_num)

#---- Compute Summary Statistics: categorical data ----#
    summary_df_cat = EDA_categorical_data(cleaned_df)
    print(summary_df_cat)

#---- Visualize response variable (salary):
    plt.figure(figsize=(14,6))
    plt.subplot(1,2,1)
    sns.boxplot(cleaned_df.salary)
    plt.subplot(1,2,2)
    sns.distplot(cleaned_df.salary, bins=20)
    plt.show()

#--- Use IQR to detect potential outliers:
    stat = cleaned_df.salary.describe()
    print(stat)
    IQR = stat['75%'] - stat['25%']
    UB = stat['75%'] + 1.5*IQR
    LB = stat['25%'] - 1.5*IQR
    print('The UB and LB for suspected salary outliers are {} and {}.'.format(LB, UB))

#--- Check LB Outeliers:
    cleaned_df.loc[cleaned_df.salary < 8.5]

#--- Check UB Outeliers:
    cleaned_df.loc[cleaned_df.salary > 220.5, 'jobType'].value_counts()

#--- Check the suspicious outliers by a position: Junior Role
    cleaned_df.loc[(cleaned_df.salary > 220.5) & (cleaned_df.jobType == 'JUNIOR')]

#---- Plot histograms ----#
    # Plot histogram: salary --> +ve skew
    EDA_plot_hist(cleaned_df, 'salary', 'green')

    # Plot histogram: yearsExperience --> +ve skew
    EDA_plot_hist(cleaned_df, 'yearsExperience', 'blue')

    # Plot histogram: milesFromMetropolis --> -ve skew
    EDA_plot_hist(cleaned_df, 'milesFromMetropolis', 'orange')
        
#---- Plot bar chart(s) ----#
    # Plot bar chart: jobType
    EDA_plot_freq_chart(cleaned_df, 'jobType')

    # Plot bar chart: degree
    EDA_plot_freq_chart(cleaned_df, 'degree')    
    
    # Plot bar chart: major
    EDA_plot_freq_chart(cleaned_df, 'major')
        
    # Plot bar chart: industry
    EDA_plot_freq_chart(cleaned_df, 'industry')
    
#---- Plot box-whisker plot chart(s) ----#
    # Plot box plot: jobType
    EDA_plot_box_whisker(cleaned_df, 'salary', 'jobType')

    # Plot box plot: degree
    EDA_plot_box_whisker(cleaned_df, 'salary', 'degree')

    # Plot box plot: major
    EDA_plot_box_whisker(cleaned_df, 'salary', 'major')
    
    # Plot box plot: major by industry
    EDA_plot_box_whisker(cleaned_df, 'salary', 'major', 'industry')

    # Plot box plot: industry
    EDA_plot_box_whisker(cleaned_df, 'salary', 'industry')
    
    # Plot box plot: industry
    EDA_plot_box_whisker(cleaned_df, 'salary', 'industry', 'jobType')

    # Drop 'jobId':
    cleaned_df.drop(list_vars, axis=1, inplace=True)

#---- Convert categorical variable data type from object to category ----#
    EDA_convert_object_to_cat(cleaned_df)

#---- Encode categorical variables using avg. salary for each category to replace label ----#
    for col in cleaned_df.columns:
        if cleaned_df[col].dtype.name == "category":
            EDA_encode_cat_var(cleaned_df, col) 

#---- Plot correlation matrix chart ----#
    # Define list of features and salary
    features = ['jobType', 'degree', 'major', 'industry', 'yearsExperience', 'milesFromMetropolis']
    label = ['salary']    

    EDA_plot_corr_matrix(cleaned_df, features, label)