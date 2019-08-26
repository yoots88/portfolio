# Import and set directory:
import os
os.getcwd()

# Set path:
#path = 'D:/P_Poverty_Prediction/A. Coding'
path = 'D:/TY Workspace/3. Data Science Portfolio/1. Completed Portfolios/p3_us_poverty/A. Coding'
os.chdir(path)

# Import all libraries #
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

# Authorization #
__author__ = "Taesun Yoo"
__email__ = "yoots1988@gmail.com"

# --- 2. Write Out List of Functions --- #
#################################
# Part 2 - Discover the Problem #
#################################
# Write a group of funtions:
def load_file(file):
    '''load input CSVs as a dataframe '''
    return pd.read_csv(file)

def join_data(df1, df2, key=None, left_index=False, right_index=False):
    '''performs inner join to return records exist in both dataframes'''
    return pd.merge(df1, df2, how='inner', on=key, left_index=left_index,
                    right_index=right_index)

def clean_data(raw_df):
    '''remove rows that contain invalid data or duplicate IDs'''
    clean_df = raw_df.drop_duplicates(subset='row_id')
#    clean_df = clean_df[clean_df['health__homicides_per_100k'] > 0]
    return clean_df

def drop_row_by_index(df, idx_list):
    df_row_drop = df.drop(df.index[idx_list])
    return df_row_drop

def drop_column_by_index(df, col_list):
    df_column_drop = df.drop([col_list], axis=1)
    return df_column_drop

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
    
def EDA_plot_hist_label(df, cat_var, bins, lab_list):
    '''split dataframe by category and plot a histogram'''
    for i in lab_list:
        df_by_label = df['poverty_rate'][df[cat_var] == i]
        plt.hist(df_by_label, bins=bins, label=i)
        plt.title('Histogram of Poverty Rate')
        plt.xlabel('Poverty Rate')
        plt.ylabel('# of US counties')                   

def EDA_plot_hist_2by2(df, var1, var2, var3, var4, factor=None):
    '''Print skewness and plot the histogram'''
    plt.figure(figsize=(8,8))
    plt.subplots_adjust(hspace=1/2, wspace=1/2)
    #subplot 1:
    print("Skewness is:" + var1, df[var1].skew())
    plt.subplot(2,2,1)
    plt.hist(df[var1]*factor, color='green')
    plt.title('Histogram of '+ var1)
    plt.xlabel(var1)
    plt.ylabel('# of US counties')
    #subplot 2:
    print("Skewness is:" + var2, df[var2].skew())
    plt.subplot(2,2,2)
    plt.hist(df[var2]*factor, color='blue')
    plt.title('Histogram of '+ var2)
    plt.xlabel(var2)
    plt.ylabel('# of US counties')
    #subplot 3:
    print("Skewness is:" + var3, df[var3].skew())
    plt.subplot(2,2,3)
    plt.hist(df[var3]*factor, color='cyan')
    plt.title('Histogram of '+ var3)
    plt.xlabel(var3)
    plt.ylabel('# of US counties')
    #subplot 4:
    print("Skewness is:" + var4, df[var4].skew())
    plt.subplot(2,2,4)
    plt.hist(df[var4]*factor, color='purple')
    plt.title('Histogram of '+ var4)
    plt.xlabel(var4)
    plt.ylabel('# of US counties')
    
def EDA_plot_freq_chart(df, cat_var):
    '''computes frequency count chart'''
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
    '''plot the box-whisker plot'''
    df.sort_values(by=[num_var, cat_var], ascending=False, inplace=True)
    plt.figure()
    sns.set(style='whitegrid')
    sns.boxplot(cat_var, num_var, hue, df)
    plt.title('Box Plot of '+ num_var + ' by '+ cat_var)
    plt.xticks(rotation=270, fontsize=9)

def EDA_convert_object_to_cat(df):
    '''convert data type object to category'''
    for col in df.columns:
        if df[col].dtype.name == "object":
            df[col] = df[col].astype('category')

def EDA_encode_cat_var(df, col):
    '''encode the categorical variables using avg. salary for each category '''
    cat_dict={}
    cats = df[col].cat.categories.tolist()
    for cat in cats:
        cat_dict[cat] = df[df[col] == cat]['poverty_rate'].mean()
    df[col] = df[col].map(cat_dict)

def EDA_plot_corr_matrix(df, features, label):
    '''plot the correlation matrix'''
    plt.figure(figsize=(12,10))
    sns.heatmap(df[features + label].corr(), cmap=sns.diverging_palette(220, 10, as_cmap=True), annot=True)
    plt.xticks(rotation=90)
    plt.show()

def EDA_plot_crosstab(df, cat_var1, cat_var2):
    '''plot a cross-tabulate on two categorical variables'''
    cross_tab = pd.crosstab(df[cat_var1], df[cat_var2])
    return cross_tab

def EDA_plot_scatter(df, var1, var2, 
                     c1, c2, factor=None):
    '''plot 2 by 1 scatter plots'''
    plt.figure(figsize=(8,8))
    plt.subplots_adjust(hspace=0.4, wspace=0.9)
    plt.subplot(2,1,1)
    plt.scatter(df[var1]*factor, df['poverty_rate'], color=c1)
    plt.title('Relationship between ' + var1 + ' and Poverty Rate')
    plt.xlabel(var1)
    plt.ylabel('Poverty Rate')

    plt.subplot(2,1,2)
    plt.scatter(df[var2]*factor, df['poverty_rate'], color=c2)
    plt.title('Relationship between '+ var2 + ' and Poverty Rate')
    plt.xlabel(var2)
    plt.ylabel('Poverty Rate')
    
def split_dataframe_by_string(df, cat_var, str_val):
    '''split dataframe by a specified string value in categorical variable'''
    df_str = df[df[cat_var].str.contains(str_val, case=True, regex=False)]
    return df_str

def EDA_plot_multi_facet_scatter(df1, df2, var1, var2, lab, factor):
    '''plot multi-faceted scatter plot by county class'''
    f, (ax1, ax2)=plt.subplots(1, 2, sharey=True, figsize=(8,4))
    plt.subplots_adjust(hspace=0.2, wspace=0.2)
    plt.tight_layout(pad=0.4, w_pad=1.5, h_pad=1.0)
    ax1.scatter(df1[var1]*factor, df1[lab], label='Nonmetro', edgecolor='w')
    ax1.scatter(df2[var1]*factor, df2[lab], label='Metro', edgecolor='w')
    ax1.legend(loc='upper right')
    ax1.set_xlabel(var1, fontsize=14)
    ax1.set_ylabel(lab, fontsize=14)
    ax1.grid(False)
    
    ax2.scatter(df1[var2]*factor, df1[lab], label='Nonmetro', edgecolor='w')
    ax2.scatter(df2[var2]*factor, df2[lab], label='Metro', edgecolor='w')
    ax2.legend(loc='upper right')
    ax2.set_xlabel(var2, fontsize=14)
    ax2.set_ylabel(lab, fontsize=14)
    ax2.grid(False)

def EDA_plot_color_sc_scatter(df, var1, var2, lab, var_color):
    '''plot color scaled scatter plots'''
    # figure 1: subplot 1
    f, (ax1, ax2) = plt.subplots(1,2, sharey=True, figsize=(7,7))
    s1 = ax1.scatter(df[var1], df[lab], c=df[var_color],
                     cmap=plt.cm.coolwarm, edgecolor='w')
    ax1.set_xlabel(var1, fontsize=14)
    ax1.set_ylabel(lab, fontsize=14)
    ax1.grid(False)
    # figure 2: subplot 2    
    ax2.scatter(df[var2], df[lab], c=df[var_color],
                     cmap=plt.cm.coolwarm, edgecolor='w')
    ax2.set_xlabel(var2, fontsize=14)
    ax2.set_ylabel(lab, fontsize=14)
    ax2.grid(False)
    # lenged: color bar scaled by confounding factor
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    cax=plt.axes([0.85, 0.1, 0.05, 0.8])
    cb=f.colorbar(s1, cax=cax)
    cb.set_label(var_color)

# --- 3. Load the data --- #
# Define input CSVs:
if __name__ == '__main__':
    eda_file = 'df_eda.csv'

# Define type of variables list:
#df_train.select_dtypes(include='object').columns
cat_vars = ['area__rucc', 'area__urban_influence', 'econ__economic_typology', 'yr']

#df_train.select_dtypes(include='int64').columns
#df_train.select_dtypes(include='float64').columns
# 'row_id', 'health__homicides_per_100k'
num_vars = ['demo__birth_rate_per_1k', 'demo__death_rate_per_1k',
                  'econ__pct_civilian_labor', 'econ__pct_unemployment', 'econ__pct_uninsured_adults', 'econ__pct_uninsured_children',
                  'demo__pct_female', 'demo__pct_below_18_years_of_age', 'demo__pct_aged_65_years_and_older', 'demo__pct_hispanic',
                  'demo__pct_non_hispanic_african_american', 'demo__pct_non_hispanic_white',
                  'demo__pct_american_indian_or_alaskan_native', 'demo__pct_asian', 'demo__pct_adults_less_than_a_high_school_diploma',
                  'demo__pct_adults_with_high_school_diploma', 'demo__pct_adults_with_some_college',
                  'demo__pct_adults_bachelors_or_higher', 'health__pct_adult_obesity', 'health__pct_adult_smoking', 'health__pct_diabetes',
                  'health__pct_low_birthweight', 'health__pct_excessive_drinking', 'health__pct_physical_inacticity',
                  'health__air_pollution_particulate_matter', 
                  'health__motor_vehicle_crash_deaths_per_100k', 'health__pop_per_dentist', 'health__pop_per_primary_care_physician']

label_var = 'poverty_rate'

# Define variables to drop
list_vars = ['row_id']

# Load data
df_eda = load_file(eda_file)

del (eda_file)

################################################
# Part 3 - Exploratory Data Analysis: Insights #
################################################
# compute top 10 rows on a eda_dataframe:
df_eda.head(10)
    
# check duplicates:
df_eda.duplicated().sum()

#---- Compute % of Missing Data ----#
missing_df = EDA_missing_data(df_eda)
print(missing_df)
    
#---- Compute Summary Statistics: numerical data ----#
summary_df_num = EDA_numerical_data(df_eda[num_vars])
print(summary_df_num)

#---- Compute Summary Statistics: categorical data ----#
summary_df_cat = EDA_categorical_data(df_eda)
print(summary_df_cat)

#---- Visualize response variable (salary) ----#
plt.figure(figsize=(14,6))
plt.subplot(1,2,1)
sns.boxplot(df_eda.poverty_rate)
plt.subplot(1,2,2)
sns.distplot(df_eda.poverty_rate, bins=20)
plt.show()

#--- Use IQR to detect potential outliers ----#
stat = df_eda.poverty_rate.describe()
print(stat)
IQR = stat['75%'] - stat['25%']
UB = stat['75%'] + 1.5*IQR
LB = stat['25%'] - 1.5*IQR
print('The LB and UB for suspected poverty rate outliers are {} and {}.'.format(LB, UB))

# Check LB Outeliers:
df_eda[df_eda.poverty_rate < 0]

# Check UB Outeliers:
df_eda[df_eda.poverty_rate > 32.75]

# check potential outliers by categorical vars:
df_eda.loc[df_eda.poverty_rate > 32.75, 'area__urban_influence'].value_counts()
df_eda.loc[df_eda.poverty_rate > 32.75, 'econ__economic_typology'].value_counts()

#--- Check the suspicious outliers by an economic typology: mining-dependent
df_eda[(df_eda.poverty_rate > 32.75) & (df_eda.econ__economic_typology == 'Mining-dependent')]

#---- Plot histograms ----#
# Create a list of economic typology:
lab_list = list(df_eda['econ__economic_typology'].unique())

# Plot multiple histograms on poverty rate by economic type:
EDA_plot_hist_label(df_eda, 'econ__economic_typology', 20, lab_list)
plt.legend()
plt.show()
    
# Plot 2by2 histogram as a subplot: demographic
EDA_plot_hist_2by2(df_eda, 'econ__pct_civilian_labor', 'econ__pct_unemployment',
                   'econ__pct_uninsured_children', 'econ__pct_uninsured_adults', 100)
plt.show()

# Plot 2by2 histogram as a subplot: health indicators
EDA_plot_hist_2by2(df_eda, 'health__pct_adult_obesity', 'health__pct_adult_smoking',
                   'health__pct_diabetes', 'health__pct_low_birthweight', 100)
plt.show()

# Plot 2by2 histogram as a subplot: education
EDA_plot_hist_2by2(df_eda, 'demo__pct_adults_less_than_a_high_school_diploma',
                  'demo__pct_adults_with_high_school_diploma', 'demo__pct_adults_with_some_college',
                  'demo__pct_adults_bachelors_or_higher', 100)
plt.show()
#---- Plot bar chart(s) ----#
# Plot bar chart: economic typology
EDA_plot_freq_chart(df_eda, 'econ__economic_typology')
plt.show()

#---- Plot box-whisker plot chart(s) ----#
# Plot box plot: economic typology
EDA_plot_box_whisker(df_eda, 'poverty_rate', 'econ__economic_typology')
plt.show()
# Plot box plot: urban degree
EDA_plot_box_whisker(df_eda, 'poverty_rate', 'area__rucc')
plt.show()
# Plot box plot: urban size
EDA_plot_box_whisker(df_eda, 'poverty_rate', 'area__urban_influence')
plt.show()

# Drop 'row_id':
#df_eda.drop(list_vars, axis=1, inplace=True)


#---- Convert categorical variable data type from object to category ----#
EDA_convert_object_to_cat(df_eda)

#---- Encode categorical variables using avg. salary for each category to replace label ----#
for col in df_eda.columns:
    if df_eda[col].dtype.name == "category":
       EDA_encode_cat_var(df_eda, col) 

#---- Plot correlation matrix chart ----#
# Define list of features and salary
features = ['demo__birth_rate_per_1k', 'demo__death_rate_per_1k',
                  'econ__pct_civilian_labor', 'econ__pct_unemployment', 'econ__pct_uninsured_adults', 'econ__pct_uninsured_children',
                  'demo__pct_female', 'demo__pct_below_18_years_of_age', 'demo__pct_aged_65_years_and_older', 'demo__pct_hispanic',
                  'demo__pct_non_hispanic_african_american', 'demo__pct_non_hispanic_white',
                  'demo__pct_american_indian_or_alaskan_native', 'demo__pct_asian', 'demo__pct_adults_less_than_a_high_school_diploma',
                  'demo__pct_adults_with_high_school_diploma', 'demo__pct_adults_with_some_college',
                  'demo__pct_adults_bachelors_or_higher', 'health__pct_adult_obesity', 'health__pct_adult_smoking', 'health__pct_diabetes',
                  'health__pct_low_birthweight', 'health__pct_excessive_drinking', 'health__pct_physical_inacticity',
                  'health__air_pollution_particulate_matter', 
                  'health__motor_vehicle_crash_deaths_per_100k', 'health__pop_per_dentist', 'health__pop_per_primary_care_physician']
label = ['poverty_rate']    

EDA_plot_corr_matrix(df_eda, features, label)
plt.show()

########################################################
# Bi-variate analyses: cross-tabulation, scatter plots #
########################################################
#---- Plot a cross-tabulate based on two categorical variables ----#    
EDA_plot_crosstab(df_eda, 'area__rucc', 'econ__economic_typology')

#---- Plot a scatter plot: numerical and categorical variables ----#
# Demographics
EDA_plot_scatter(df_eda, 'econ__pct_civilian_labor', 'econ__pct_uninsured_adults',
                 'red', 'blue', 100)
plt.show()

# Health indicators
EDA_plot_scatter(df_eda, 'health__pct_excessive_drinking', 'health__pct_low_birthweight',
                 'red', 'blue', 100)
plt.show()

# Education indicators
EDA_plot_scatter(df_eda, 'demo__pct_adults_bachelors_or_higher', 'demo__pct_adults_less_than_a_high_school_diploma',
                 'red', 'blue', 100)
plt.show()

#---- Plot multi-faceted scatter plots by categorical variable ----#
df_non_metro = split_dataframe_by_string(df_eda, 'area__rucc', 'Nonmetro')
df_metro = split_dataframe_by_string(df_eda, 'area__rucc', 'Metro')


EDA_plot_multi_facet_scatter(df_non_metro, df_metro, 
                             'demo__pct_adults_less_than_a_high_school_diploma',
                             'demo__pct_adults_bachelors_or_higher', 'poverty_rate', 100)
plt.show()

#---- Plot color scaled scatter plots by numerical variable ----#
EDA_plot_color_sc_scatter(df_eda, 'demo__pct_adults_less_than_a_high_school_diploma',
                          'demo__pct_adults_bachelors_or_higher', 'econ__pct_civilian_labor',
                          'poverty_rate')
plt.show()