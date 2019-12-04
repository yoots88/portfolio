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
    '''load the input file CSVs'''
    df = pd.read_csv(file)
    return df

def clean_data(raw_df, col_name):
    '''drop the duplicate rows based on specified column'''
    cleand_df = raw_df.drop_duplicates(subset=col_name)
    return cleand_df

def EDA_drop_column(df, var_list):
    '''drop column(s) by a list'''
    df.drop(var_list, axis=1, inplace=True)
    return df

def EDA_convert_dt_type(df, var_name):
    '''Convert data type as an object'''
    df[var_name] = df[var_name].astype('object')

def EDA_convert_object_to_cat(df):
    '''convert object to category data types'''
    for col in df.columns:
        if df[col].dtype.name == 'object':
            df[col] = df[col].astype('category')

def EDA_missing_data(cleaned_df):
    '''performs the missing % on dataframe'''
    missing_df = pd.DataFrame(cleaned_df.isnull().sum())
    missing_df.columns = ['count']
    missing_df['pct'] = (missing_df['count']/len(cleaned_df))*100
    return missing_df

def EDA_stat_numerical(cleaned_df):
    '''performs the summary statistics on numerical vars'''
    summary_df_num = cleaned_df.describe(include='float64').T
    summary_df_num = pd.DataFrame(summary_df_num)[['count','std','min','mean','max','25%','50%','75%']]
    return summary_df_num

def EDA_stat_categorical(cleaned_df):
    '''performs the summary statistics on categorical vars'''
    summary_df_cat = cleaned_df.describe(include='O').T
    summary_df_cat = pd.DataFrame(summary_df_cat)
    return summary_df_cat

def EDA_compute_IQR(df, var_name):
    '''Compute the Interquartile Range and print the outliers'''
    df['IQR'] = df['75%'] - df['25%']
    df['LB'] = df['25%'] - 1.5*df['IQR']
    df['UB'] = df['75%'] + 1.5*df['IQR']
    print('The LB and UB for suspected '+var_name+' outliers are {} and {}.'.format(df['LB'].loc[var_name], df['UB'].loc[var_name]))
    return df

def EDA_plot_num_var(df, num_var):
    '''plot a boxplot and distribution plot for numerical variables'''
    plt.figure(figsize=(14,7))
    plt.subplot(1,2,1)
    sns.boxplot(df[num_var])
    plt.subplot(1,2,2)
    sns.distplot(df[num_var], bins=20)
    plt.show()
    
def EDA_plot_hist_label(df, num_var, cat_var, bins, lab_list):
    '''split a dataframe by categories and plot a histogram'''
    for i in lab_list:
        df_by_label = df[num_var][df[cat_var]==i]
        plt.hist(df_by_label, bins=bins, label=i)
        plt.title('Histogram of ' + str(num_var))
        plt.xlabel(str(num_var))
        plt.ylabel('Weekly Transactions')

def EDA_plot_hist_2by2(df, var1, bin1,
                       var2, bin2,
                       var3, bin3,
                       var4, bin4, factor=None):
    '''Print skewness and plot the histograms'''
    plt.figure(figsize=(10,10))
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    #subplot 1
    print("skewness is:" +var1, df[var1].skew())
    plt.subplot(2,2,1)
    plt.hist(df[var1]*factor, color='green', bins=bin1)
    plt.title('Histogram of '+var1)
    plt.xlabel(var1)
    plt.ylabel('Weekly Transactions')
    #subplot 2
    print("skewness is:" +var2, df[var2].skew())
    plt.subplot(2,2,2)
    plt.hist(df[var2]*factor, color='blue', bins=bin2)
    plt.title('Histogram of '+var2)
    plt.xlabel(var2)
    plt.ylabel('Weekly Transactions')
    #subplot 3
    print("skewness is:" +var3, df[var3].skew())
    plt.subplot(2,2,3)
    plt.hist(df[var3]*factor, color='orange', bins=bin3)
    plt.title('Histogram of '+var3)
    plt.xlabel(var3)
    plt.ylabel('Weekly Transactions')
    #subplot 4
    print("skewness is:" +var4, df[var4].skew())
    plt.subplot(2,2,4)
    plt.hist(df[var4]*factor, color='purple', bins=bin4)
    plt.title('Histogram of '+var4)
    plt.xlabel(var4)
    plt.ylabel('Weekly Transactions')
 
def split_data_by_label(df, lab):
    '''set label as an index to split dataframes'''
    df_label = df.set_index(lab)
    df_label_0, df_label_1 = df_label.loc[0], df_label.loc[1]
    return (df_label_0, df_label_1)

def group_data_by_label(df, lab, cat_var):
    df_grp = pd.DataFrame(df.groupby([lab, cat_var])[cat_var].count())
    df_grp.columns = ['count']
    df_grp_0 = df_grp.loc[0]
    df_grp_1 = df_grp.loc[1]
    return (df_grp_0, df_grp_1)

def plot_pie_charts(df_0, df_1, label, var_name):
    '''Plot a series of pie charts by a label'''
    f, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))
    plt.subplots_adjust(hspace=.5, wspace=.5)
    # subplot for non group label:
    ax1.pie(df_0, autopct='%.0f%%',
            wedgeprops={'edgecolor':'white'}, textprops={'fontsize':14})
    ax1.set_title('Non-Holiday Weekly Transactions '+ 'by '+ var_name, fontsize=13)
    ax1.legend(labels = label, loc='upper right')
    # subplot for group label:
    ax2.pie(df_1, autopct='%.0f%%',
            wedgeprops={'edgecolor':'white'}, textprops={'fontsize':14})
    ax2.set_title('Holiday Weekly Transactions '+ 'by '+ var_name, fontsize=13)
    ax2.legend(labels = label, loc='upper right')

def EDA_plot_freq_chart(df, cat_var):
    '''computes frequency chart'''
    plt.figure(figsize=(15,15))
    var_count = df[cat_var].value_counts()
    sns.barplot(var_count.index, var_count.values, order=var_count.index, alpha=0.9)
    plt.title('Frequency counts of '+ cat_var)
    plt.ylabel('# of Weekly Transactions')
    plt.xlabel(cat_var, fontsize=10)
    plt.xticks(rotation=270)
    plt.show()

def EDA_plot_mean_cat_chart(df, num_var, cat_var1, cat_var2):
    '''plot 2by1 avg. label bar charts based on categorical vars'''
    plt.figure(figsize=(10,10))
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    # plot 1
    plt.subplot(2,1,1)
    df.groupby(cat_var1)[num_var].mean().sort_values(ascending=False).plot(kind='bar')
    plt.xlabel(cat_var1)
    plt.ylabel('Mean '+ num_var)
    plt.title('Mean ' + num_var + ' by '+ cat_var1)
    # plot 2
    plt.subplot(2,1,2)
    df.groupby(cat_var2)[num_var].mean().sort_values(ascending=False).plot(kind='bar')
    plt.xlabel(cat_var2)
    plt.ylabel('Mean '+ num_var)
    plt.title('Mean ' + num_var + ' by '+ cat_var2)    

def EDA_plot_box(df, num_var, cat_var, hue=None):
    '''plot the box-whisker'''
    df.sort_values(by=[num_var, cat_var], ascending=False, inplace=True)
    plt.figure()
    sns.set(style='whitegrid')
    sns.boxplot(cat_var, num_var, hue, df)
    plt.title('Box plot of '+ num_var + ' by '+ cat_var)
    plt.xticks(rotation=270, fontsize=9)
    
def EDA_plot_corr_matrix(df):
    '''plot correlation matrix'''
    cm = df[features + label].corr()
    mask = np.zeros_like(cm)
    mask[np.triu_indices_from(mask)]=True
    plt.figure(figsize=(10,10))
    sns.heatmap(cm, cmap=sns.diverging_palette(220, 10, as_cmap=True),
                annot=True, fmt='.2f', mask=mask)
    plt.xticks(rotation=90)
    plt.show()
    
def EDA_plot_crosstab(df, cat_var1, cat_var2):
    ct = pd.crosstab(df[cat_var1], df[cat_var2])
    df_ct = pd.DataFrame(ct)
    return df_ct

def EDA_plot_scatter(df, lab, var1, c1, 
                     var2, c2, factor=None):
    '''plot scatter plots 2by1'''
    plt.figure(figsize=(10,10))
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    # plot 1
    plt.subplot(2,1,1)
    plt.scatter(df[var1]*factor, df[lab], color=c1)
    plt.title('Relationship between '+ lab + ' and ' + var1)
    plt.xlabel(var1)
    plt.ylabel(lab)
    # plot 2
    plt.subplot(2,1,2)
    plt.scatter(df[var2]*factor, df[lab], color=c2)
    plt.title('Relationship between '+ lab + ' and ' + var2)
    plt.xlabel(var2)
    plt.ylabel(lab)

def EDA_plot_multi_facet_scatter(df_1, df_2, var_x, var_y):
    '''plot multi-faceted scatter plot by a categorical factor'''
    plt.subplots_adjust(hspace=.2, wspace=.2)
    plt.scatter(df_1[var_x], df_1[var_y], label='Holiday', edgecolor='w')
    plt.scatter(df_2[var_x], df_2[var_y], label='Non-Holiday', edgecolor='w')
    plt.legend(loc='upper right')
    plt.grid(False)
    plt.xlabel(var_x, fontsize=14)
    plt.ylabel(var_y, fontsize=14)
    plt.show()

def EDA_plot_color_sc_scatter(df, var_x, var_y, lab):
    '''plot color scaled scatter plot(s) by a label'''
    # subplot 1
    plt.figure(figsize=(10,6))
    s1 = plt.scatter(df[var_x], df[var_y], c=df[lab],
                     cmap=plt.cm.coolwarm, edgecolor='w')
    plt.xlabel(var_x, fontsize=14)
    plt.ylabel(var_y, fontsize=13)
    plt.grid(False)
    # legend: color bar scaled by label
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    cax=plt.axes([0.85, 0.1, 0.05, 0.8])
    cbar=plt.colorbar(s1, cax=cax)
    cbar.set_label(lab)
    
# --- 3. Load the data --- #
# Define input CSVs:
if __name__ == '__main__':
    EDA_file = 'df_eda.csv'

# Load data
df_cleaned = load_file(EDA_file)

# Define type of variables list:
#df_train.select_dtypes(include='object').columns

################################################
# Part 3 - Exploratory Data Analysis: Insights #
################################################
# compute top 10 rows on a eda_dataframe:
df_cleaned.head()    

# convert data types as object: Dept and Store
EDA_convert_dt_type(df_cleaned, 'Dept')
EDA_convert_dt_type(df_cleaned, 'Store')

# variable list to drop:
var_list = ['File_Type', 'md1_presence', 'md2_presence', 'md3_presence', 'md4_presence', 'md5_presence',
            'Date_String', 'Month', 'IsHoliday', 'Black_Friday', 'Christmas_Eves']

# drop column by index:
df_cleaned = EDA_drop_column(df_cleaned, var_list)
df_cleaned.dtypes

#---- Compute % of Missing Data ----#
df_missing = EDA_missing_data(df_cleaned)
df_missing
    
#---- Compute Summary Statistics: numerical data ----#
df_stat_num = EDA_stat_numerical(df_cleaned)
df_stat_num

#---- Compute Summary Statistics: categorical data ----#
df_stat_cat = EDA_stat_categorical(df_cleaned)
df_stat_cat

#---- Visualize response variable (weekly_sales) ----#
EDA_plot_num_var(df_cleaned, 'Weekly_Sales')

#--- Use IQR to detect potential outliers ----#
df_stat_num = EDA_compute_IQR(df_stat_num, 'Weekly_Sales')
# IQR = Q3 - Q1, UB = Q3 + 1.5*IQR, LB = Q1 - 1.5*IQR
    
# Check LB & UB Outliers:
df_cleaned[df_cleaned['Weekly_Sales'] < -25110]
df_cleaned[df_cleaned['Weekly_Sales'] > 47395]

# UB outliers of 35,521 Weekly Sales record over $47,395
df_outliers_UB = df_cleaned[df_cleaned['Weekly_Sales'] > 47395]

# check potential outliers by categorical vars:
# Top sales occured for Dept = 38, Store = 4, Type = A 
df_outliers_UB.describe(include='O').T

#--- Check the suspicious outliers by a store: department and type
df_outliers_UB[(df_outliers_UB['Store'] == 4) & (df_outliers_UB['Dept'] == 38) & (df_outliers_UB['Type'] == 'A')]

#---- Plot histograms ----#
# Create a list of type
lab_list = list(df_cleaned['Type'].unique())

# Plot multiple histograms on Weekly_Sales by Type:
EDA_plot_hist_label(df_cleaned, 'Weekly_Sales', 'Type', 20, lab_list)
plt.legend()
plt.show()
    
# Plot 2by2 histogram as a subplot: demographic    
EDA_plot_hist_2by2(df_cleaned, 'Temperature', 15,
                   'Fuel_Price', 15,
                   'CPI', 15, 'Unemployment', 15, 1)
plt.show()

df_cleaned.columns

################################
# Data Transformation: SplitBy #
################################
# Split dataframes: IsHoliday Yes/No    
df_holiday_0, df_holiday_1 = split_data_by_label(df_cleaned, 'IsHoliday_True')

# Split dataframes: Black_Friday Yes/No    
#df_Black_Friday_0, df_Black_Friday_1 = split_data_by_label(df_cleaned, 'Black_Friday_yes')

# Split dataframes: Christmas_Eves Yes/No    
#df_Christmas_Eves_0, df_Christmas_Eves_1 = split_data_by_label(df_cleaned, 'Christmas_Eves_yes')

#####################################
# Data Transformation: GroupBy Type #
#####################################
df_grp_Type_0, df_grp_Type_1 = group_data_by_label(df_cleaned, 'IsHoliday_True', 'Type')

# Plot pie charts:    
plot_pie_charts(df_grp_Type_0, df_grp_Type_1, df_grp_Type_0.index, 'Type')

#---- Plot bar/frequency chart(s) ----#
# Compute freq. chart by Store    
EDA_plot_freq_chart(df_cleaned, 'Store')

# Compute freq. chart by Department
EDA_plot_freq_chart(df_cleaned, 'Dept')

# Plot avg. label bar charts by categorical vars

# Compute 2by1 avg. weekly sales by Type and Size:    
EDA_plot_mean_cat_chart(df_cleaned, 'Weekly_Sales', 'Type', 'Size')

# Compute 2by1 avg. weekly sales by Store and Dept:
EDA_plot_mean_cat_chart(df_cleaned, 'Weekly_Sales', 'Store', 'Dept')

#---- Plot box-whisker plot chart(s) ----#
# Plot box plot: 
EDA_plot_box(df_cleaned, 'Weekly_Sales', 'Type')

EDA_plot_box(df_cleaned, 'Weekly_Sales', 'Dept')

EDA_plot_box(df_cleaned, 'Weekly_Sales', 'Store')

#---- Convert categorical variable data type from object to category ----#
EDA_convert_object_to_cat(df_cleaned)

#---- Encode categorical variables using avg. salary for each category to replace label ----#
df_cleaned.columns

#---- Plot correlation matrix chart ----#
# Define list of features and label
features = ['Temperature', 'Fuel_Price', 'Size', 'MarkDown1',
            'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 'CPI',
            'Unemployment', 'IsHoliday_True', 'Black_Friday_yes' , 'Christmas_Eves_yes',
            'Lagged_1wk_Sales', 'Sales_Var_Median_Lagged', 'Sales_Var_Median_Weekly']
label = ['Weekly_Sales']
    
EDA_plot_corr_matrix(df_cleaned)
########################################################
# Bi-variate analyses: cross-tabulation, scatter plots #
########################################################
#---- Plot a cross-tabulate based on two categorical variables ----#    
df_ct_1 = EDA_plot_crosstab(df_cleaned, 'Type', 'Dept')
df_ct_1

df_ct_2 = EDA_plot_crosstab(df_cleaned, 'Type', 'Store')
df_ct_2

#---- Plot a scatter plot: numerical and categorical variables ----#        
EDA_plot_scatter(df_cleaned, 'Weekly_Sales', 'Temperature', 'red', 
                 'Fuel_Price', 'blue', 1)

EDA_plot_scatter(df_cleaned, 'Weekly_Sales', 'Unemployment', 'purple', 
                 'CPI', 'green', 1)

EDA_plot_scatter(df_cleaned, 'Weekly_Sales', 'Lagged_1wk_Sales', 'orange', 
                 'Size', 'black', 1)

#---- Plot multi-faceted scatter plots by categorical variable ----#    
EDA_plot_multi_facet_scatter(df_holiday_0, df_holiday_1, 
                             'Size', 'Weekly_Sales')


#---- Plot color scaled scatter plots by numerical variable ----#
EDA_plot_color_sc_scatter(df_cleaned, 'Size', 'Weekly_Sales', 'CPI')
EDA_plot_color_sc_scatter(df_cleaned, 'Size', 'Weekly_Sales', 'Unemployment')


df_cleaned.columns
############################################################
# Time-Series: Analyses with Specific Store, Dept and Type # 
############################################################
# Create a dataframe index by sales date:
# Subset sales for Store=4
df_store_4 = df_cleaned[df_cleaned['Store']==4]
df_store_4.dtypes

# Set the Date column as dateindex type:
df_store_4['Date'] = pd.to_datetime(df_store_4['Date'])
df_store_4= df_store_4.set_index('Date')
df_store_4.sort_index(axis=0)

# Group them by similar types of numerical features
cols_markdown = ['MarkDown1', 'MarkDown5']
cols_sales = ['Weekly_Sales', 'Lagged_1wk_Sales']

# Plot: daily time-series - MarkDown
df_store_4.loc[:, cols_markdown].plot(figsize=(10,5), linewidth=5, fontsize=10, colormap='Dark2')
plt.xlabel('Sales Date', fontsize=10)
plt.show()

# Plot: daily time-series - weekly sales
df_store_4.loc[:, cols_sales].plot(figsize=(10,5), linewidth=5, fontsize=10, colormap='Dark2')
plt.xlabel('Sales Date', fontsize=10)
plt.show()

##########################
# Resampling Sales Data #
#########################
df_weekly_sales_4 = df_store_4.resample('D').first().interpolate()

# Check missing time points:
print(df_weekly_sales_4.isnull().sum())

# Downsampling to monthy by aggregation (mean):
df_monthly_sales_4 = df_weekly_sales_4.resample('M').mean()

# Rolling mean - extract using [[]] double brackets
weekly_sales = df_monthly_sales_4[['Weekly_Sales']]
lagged_weekly_sales = df_monthly_sales_4[['Lagged_1wk_Sales']]
CPI = df_monthly_sales_4[['CPI']]
Unemployment = df_monthly_sales_4[['Unemployment']]
Fuel_Price = df_monthly_sales_4[['Fuel_Price']]
MarkDown5 = df_monthly_sales_4[['MarkDown5']]
Temp = df_monthly_sales_4[['Temperature']]

# Plot: Rolling Mean over 12 months
Unemployment.rolling(12).mean().plot(figsize=(10,5), linewidth=5, fontsize=15, color='navy')
plt.xlabel('Sales Year', fontsize=15)
plt.show()

CPI.rolling(12).mean().plot(figsize=(10,5), linewidth=5, fontsize=15, color='crimson')
plt.xlabel('Sales Year', fontsize=15)
plt.show()

# Plot rolling mean of weekly_sales and unemployment together:
df_group_1 = pd.concat([weekly_sales, lagged_weekly_sales], axis=1)
df_group_1.rolling(12).mean().plot(figsize=(10,5), linewidth=5, fontsize=15)
plt.xlabel('Sales Year', fontsize=14)
plt.show()

# Plot first order difference:
Unemployment.diff().plot(figsize=(10,5), linewidth=5, fontsize=15, color='navy')
plt.xlabel('Sales Year', fontsize=14)
plt.show()

# Plot rolling mean of weekly_sales and unemployment together:
df_diff_group_1 = pd.concat([weekly_sales, lagged_weekly_sales], axis=1)
df_diff_group_1.diff().plot(figsize=(10,5), linewidth=5, fontsize=15)
plt.xlabel('Sales Year', fontsize=14)
plt.show()

# Correlation Matrix: 
cols_grp_1 = ['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5','IsHoliday_True']


# Groupby IsHoliday: Holiday vs. Non-Holiday Weeks
df_store_4.loc[:, cols_grp_1].groupby('IsHoliday_True').corr()


# Plot: auto-correlation
pd.plotting.autocorrelation_plot(weekly_sales)
plt.title('Auto-correlation: Weekly Sales')
plt.show()